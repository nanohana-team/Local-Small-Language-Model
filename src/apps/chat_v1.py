from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict

from src.core.convergence import run_convergence_v1
from src.core.divergence import run_divergence_v1
from src.core.io.lsd_lexicon import (
    inspect_lexicon_storage,
    load_indexed_lsd_lexicon_container,
    load_lexicon_container,
    profile_lexicon_load,
)
from src.core.evaluation import (
    apply_external_feedback,
    build_external_signal,
    summarize_external_result,
    summarize_teacher_result,
)
from src.core.logging import (
    TraceLogger,
    build_teacher_output_record,
    build_teacher_request_record,
    build_teacher_selection_record,
)
from src.core.records import EpisodeWriter, ImprovementCandidateWriter, build_teacher_improvement_candidate
from src.core.scoring import score_turn_v1
from src.core.planning import build_plan_v1
from src.core.relation import RelationIndex, build_relation_index, validate_relation_graph
from src.core.slotting import fill_slots_v1
from src.core.surface import render_surface_v1
from src.llm import ExternalTeacherOrchestrator, TeacherTurnRequest

STARTUP_CACHE_VERSION = 2


class MinimalChatEngine:
    """LSLM v4 minimal vertical slice.

    The engine intentionally stays small: it validates the relation graph,
    extracts seed concepts, performs relation-based divergence/convergence,
    fills a minimal slot structure, and renders 1–2 sentences.
    """

    def __init__(
        self,
        lexicon_path: str | Path,
        *,
        runtime_dir: str | Path = "runtime",
        trace_mode: str = "standard",
        strict_schema: bool = False,
        strict_relations: bool = True,
        require_closed_relations: bool = True,
        startup_mode: str = "auto",
        startup_cache: bool = True,
        rebuild_startup_cache: bool = False,
        enable_external_eval: bool = False,
        enable_external_teacher: bool = False,
        llm_order_path: str | Path = "settings/LLM_order.yaml",
        teacher_profile_path: str | Path = "settings/teacher_profiles.yaml",
        scoring_config_path: str | Path = "settings/scoring_v1.yaml",
    ) -> None:
        self.lexicon_path = Path(lexicon_path)
        self.runtime_dir = Path(runtime_dir)
        self.logger = TraceLogger(self.runtime_dir, mode=trace_mode)
        self.startup_info: Dict[str, Any] = {}
        self.enable_external_eval = enable_external_eval
        self.enable_external_teacher = enable_external_teacher
        self.scoring_config_path = str(scoring_config_path)
        self.episode_writer = EpisodeWriter(self.runtime_dir) if (enable_external_eval or enable_external_teacher) else None
        self.improvement_writer = ImprovementCandidateWriter(self.runtime_dir) if enable_external_teacher else None
        self.external_orchestrator = None
        if enable_external_eval or enable_external_teacher:
            self.external_orchestrator = ExternalTeacherOrchestrator(
                llm_order_path=llm_order_path,
                teacher_profile_path=teacher_profile_path,
            )

        started = time.perf_counter()
        storage = inspect_lexicon_storage(self.lexicon_path)
        self.startup_info["storage"] = storage

        use_fast_indexed = self._should_use_fast_indexed(storage, startup_mode)
        cache_dir = self.runtime_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._startup_cache_path(cache_dir, storage, strict_schema, strict_relations, require_closed_relations)

        self.container: Dict[str, Any]
        self.index: RelationIndex
        self.validation_report: Dict[str, Any]

        loaded_from_cache = False
        if use_fast_indexed and startup_cache and not rebuild_startup_cache:
            cached_payload = self._load_startup_cache(cache_path)
            if cached_payload is not None:
                loaded_from_cache = True
                self.index = cached_payload["relation_index"]
                self.validation_report = cached_payload["validation_report"]
                self.container = {
                    "meta": cached_payload.get("meta", {}),
                    "concepts": self.index.concepts,
                    "slot_frames": self.index.slot_frames,
                    "indexes": {
                        "concept_to_entries": self.index.concept_to_entries,
                        "surface_to_entry": self.index.surface_to_entries,
                    },
                }
                self.startup_info["startup_path"] = "indexed_cache"
                self.startup_info["cache_path"] = str(cache_path)

        if not loaded_from_cache:
            load_started = time.perf_counter()
            if use_fast_indexed:
                self.container = load_indexed_lsd_lexicon_container(
                    self.lexicon_path,
                    normalize=False,
                    lightweight=True,
                )
                self.startup_info["startup_path"] = "indexed_lightweight"
            else:
                self.container = load_lexicon_container(self.lexicon_path)
                self.startup_info["startup_path"] = "full_container"
            self.startup_info["container_load_ms"] = round((time.perf_counter() - load_started) * 1000.0, 3)

            index_started = time.perf_counter()
            self.index = build_relation_index(self.container)
            self.startup_info["relation_index_ms"] = round((time.perf_counter() - index_started) * 1000.0, 3)

            validation_started = time.perf_counter()
            self.validation_report = validate_relation_graph(
                self.container,
                strict_schema=strict_schema,
                strict_relations=strict_relations,
                require_closed_relations=require_closed_relations,
                prebuilt_index=self.index,
            )
            self.startup_info["validation_ms"] = round((time.perf_counter() - validation_started) * 1000.0, 3)

            if use_fast_indexed and startup_cache:
                self._save_startup_cache(
                    cache_path,
                    {
                        "version": STARTUP_CACHE_VERSION,
                        "relation_index": self.index,
                        "validation_report": self.validation_report,
                        "meta": dict(self.container.get("meta", {})),
                    },
                )
                self.startup_info["cache_path"] = str(cache_path)
                self.startup_info["cache_saved"] = True

        if not self.validation_report.get("ok", False):
            preview = "\n".join(self.validation_report.get("errors", [])[:20])
            raise ValueError(f"Relation validation failed:\n{preview}")

        self.startup_info["total_startup_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        self.logger.info(
            "minimal_chat_engine_ready "
            f"startup_path={self.startup_info.get('startup_path')} "
            f"concepts={len(self.index.concepts)} lexical_entries={len(self.index.lexical_entries)} "
            f"startup_ms={self.startup_info['total_startup_ms']:.3f}"
        )
        for warning in self.validation_report.get("warnings", [])[:20]:
            self.logger.warning(f"relation_validation_warning {warning}")

    @staticmethod
    def _should_use_fast_indexed(storage: Dict[str, Any], startup_mode: str) -> bool:
        if startup_mode == "full":
            return False
        if startup_mode == "fast":
            return storage.get("storage") == "indexed_lsdx"
        return storage.get("storage") == "indexed_lsdx"

    def _startup_cache_path(
        self,
        cache_dir: Path,
        storage: Dict[str, Any],
        strict_schema: bool,
        strict_relations: bool,
        require_closed_relations: bool,
    ) -> Path:
        stat = self.lexicon_path.stat()
        digest = hashlib.sha256(
            json.dumps(
                {
                    "path": str(self.lexicon_path.resolve()),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "storage": storage.get("storage"),
                    "strict_schema": strict_schema,
                    "strict_relations": strict_relations,
                    "require_closed_relations": require_closed_relations,
                    "cache_version": STARTUP_CACHE_VERSION,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:16]
        return cache_dir / f"{self.lexicon_path.stem}.{digest}.startup.pkl"

    @staticmethod
    def _load_startup_cache(path: Path) -> Dict[str, Any] | None:
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
        except FileNotFoundError:
            return None
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("version") != STARTUP_CACHE_VERSION:
            return None
        relation_index = payload.get("relation_index")
        validation_report = payload.get("validation_report")
        if not isinstance(relation_index, RelationIndex):
            return None
        if not isinstance(validation_report, dict):
            return None
        return payload

    @staticmethod
    def _save_startup_cache(path: Path, payload: Dict[str, Any]) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(path)

    def run_turn(self, text: str) -> Dict[str, Any]:
        turn_id = self.logger.next_turn_id()
        started = time.perf_counter()

        divergence_started = time.perf_counter()
        provisional_divergence = run_divergence_v1(text, build_plan_v1(text, topic_count=0), self.index)
        initial_topic_count = len(provisional_divergence.input_features.get("seed_concepts", []))
        unknown_word_count = len(provisional_divergence.input_features.get("unknown_words", []))
        plan_started = time.perf_counter()
        plan = build_plan_v1(text, topic_count=initial_topic_count, unknown_words=unknown_word_count)
        plan_ms = (time.perf_counter() - plan_started) * 1000.0

        divergence = run_divergence_v1(text, plan, self.index)
        divergence_ms = (time.perf_counter() - divergence_started) * 1000.0

        convergence_started = time.perf_counter()
        convergence = run_convergence_v1(divergence, plan, self.index)
        convergence_ms = (time.perf_counter() - convergence_started) * 1000.0

        slot_started = time.perf_counter()
        slots = fill_slots_v1(plan, divergence, convergence, self.index)
        slot_ms = (time.perf_counter() - slot_started) * 1000.0

        surface_started = time.perf_counter()
        surface = render_surface_v1(plan, slots, accepted_relations=convergence.accepted_relations)
        surface_ms = (time.perf_counter() - surface_started) * 1000.0

        total_ms = (time.perf_counter() - started) * 1000.0

        turn_score = score_turn_v1(
            plan=plan,
            divergence=divergence,
            convergence=convergence,
            slots=slots,
            response_text=surface["final_text"],
            total_ms=total_ms,
            validation_report=self.validation_report,
            scoring_config_path=self.scoring_config_path,
        )
        scores = turn_score.scores
        reward = turn_score.reward
        feedback = turn_score.feedback
        score_details = turn_score.details

        evaluator_summary = None
        teacher_summary = None
        evaluator_ms = 0.0
        external_signal = None

        trace = {
            "session_id": self.logger.session_id,
            "turn_id": turn_id,
            "input": text,
            "input_features": divergence.input_features,
            "plan": plan.to_dict(),
            "divergence_candidates": [candidate.to_dict() for candidate in divergence.candidate_concepts[:12]],
            "explored_relations": divergence.explored_relations,
            "convergence_candidates": convergence.accepted_concepts,
            "accepted_relations": convergence.accepted_relations,
            "filled_slots": slots.filled_slots,
            "missing_slots": slots.missing_slots,
            "surface_plan": surface["sentence_plan"],
            "response": surface["final_text"],
            "scores": scores,
            "reward": reward,
            "feedback": feedback,
            "score_details": score_details,
            "timing": {
                "total_ms": round(total_ms, 3),
                "input_analysis_ms": round(0.0, 3),
                "plan_ms": round(plan_ms, 3),
                "divergence_ms": round(divergence_ms, 3),
                "convergence_ms": round(convergence_ms, 3),
                "slot_ms": round(slot_ms, 3),
                "surface_ms": round(surface_ms, 3),
                "evaluator_ms": 0.0,
            },
            "startup": self.startup_info,
            "teacher_requests": [],
            "teacher_outputs": [],
            "teacher_selection": None,
            "teacher_improvement_candidates": [],
        }

        if self.external_orchestrator is not None and (self.enable_external_eval or self.enable_external_teacher):
            external_payload = {
                "input": trace["input"],
                "plan": trace["plan"],
                "accepted_relations": trace["accepted_relations"],
                "filled_slots": trace["filled_slots"],
                "response": trace["response"],
                "scores": trace["scores"],
                "reward": trace["reward"],
                "feedback": trace["feedback"],
                "score_details": trace["score_details"],
            }
            if self.enable_external_eval:
                external_started = time.perf_counter()
                evaluator_result = self.external_orchestrator.evaluate_turn(external_payload)
                evaluator_ms = (time.perf_counter() - external_started) * 1000.0
                evaluator_summary = summarize_external_result(evaluator_result)
                trace["external_evaluator"] = evaluator_summary
                trace["timing"]["evaluator_ms"] = round(evaluator_ms, 3)

            if self.enable_external_teacher:
                teacher_request = TeacherTurnRequest.from_turn_payload(external_payload)
                teacher_payload = teacher_request.to_payload()
                teacher_result = self.external_orchestrator.teach_turn(teacher_payload)
                teacher_summary = summarize_teacher_result(teacher_result)
                trace["external_teacher"] = teacher_summary
                trace["teacher_requests"] = [
                    build_teacher_request_record(teacher_payload, mode="teacher")
                ]
                trace["teacher_outputs"] = [
                    build_teacher_output_record(teacher_summary)
                ]
                trace["teacher_selection"] = build_teacher_selection_record(teacher_summary)

            external_signal = build_external_signal(
                evaluator_summary,
                teacher_summary,
                scoring_config_path=self.scoring_config_path,
            )
            reward = apply_external_feedback(
                reward,
                evaluator_summary=evaluator_summary,
                teacher_summary=teacher_summary,
                scoring_config_path=self.scoring_config_path,
            )
            trace["reward"] = reward
            trace["external_signal"] = external_signal

            improvement_candidate = build_teacher_improvement_candidate(
                session_id=self.logger.session_id,
                turn_id=turn_id,
                user_input=trace["input"],
                response=trace["response"],
                plan=trace["plan"],
                teacher_summary=teacher_summary,
                external_signal=external_signal,
            )
            if improvement_candidate is not None:
                trace["teacher_improvement_candidates"] = [improvement_candidate]
                if self.improvement_writer is not None:
                    self.improvement_writer.write(improvement_candidate)

            if self.episode_writer is not None:
                self.episode_writer.write(
                    {
                        "session_id": self.logger.session_id,
                        "turn_id": turn_id,
                        "input": trace["input"],
                        "plan": trace["plan"],
                        "accepted_relations": trace["accepted_relations"],
                        "filled_slots": trace["filled_slots"],
                        "response": trace["response"],
                        "scores": trace["scores"],
                        "reward": trace["reward"],
                        "feedback": trace["feedback"],
                        "score_details": trace["score_details"],
                        "external_signal": trace.get("external_signal"),
                        "external_evaluator": evaluator_summary,
                        "external_teacher": teacher_summary,
                        "teacher_requests": trace.get("teacher_requests", []),
                        "teacher_outputs": trace.get("teacher_outputs", []),
                        "teacher_selection": trace.get("teacher_selection"),
                        "teacher_improvement_candidates": trace.get("teacher_improvement_candidates", []),
                    }
                )
        self.logger.debug(
            f"turn_completed turn_id={turn_id} accepted_concepts={len(convergence.accepted_concepts)} total_ms={total_ms:.3f}"
        )
        self.logger.record_trace(trace)
        return trace



def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LSLM v4 minimal relation-first chat pipeline.")
    parser.add_argument(
        "--lexicon",
        default="runtime/dictionaries/bootstrapped_v1.json",
        help="Path to the JSON/LSD/LSDX lexicon container.",
    )
    parser.add_argument("--text", help="Single-turn input text. If omitted, starts interactive mode.")
    parser.add_argument("--runtime-dir", default="runtime", help="Runtime directory for logs and traces.")
    parser.add_argument(
        "--trace-mode",
        default="standard",
        choices=["minimal", "standard", "deep_trace"],
        help="Trace verbosity mode.",
    )
    parser.add_argument("--allow-open-relations", action="store_true", help="Do not require a closed relation graph.")
    parser.add_argument("--non-strict-schema", action="store_true", help="Relax top-level schema validation.")
    parser.add_argument("--dump-trace", action="store_true", help="Print the full trace JSON instead of only the response.")
    parser.add_argument(
        "--startup-mode",
        default="auto",
        choices=["auto", "fast", "full"],
        help="Startup path. 'auto' uses indexed lightweight load for .lsdx, 'fast' forces it when possible, 'full' always materializes the full container.",
    )
    parser.add_argument("--no-startup-cache", action="store_true", help="Disable the startup cache for indexed lexicons.")
    parser.add_argument("--rebuild-startup-cache", action="store_true", help="Ignore and rebuild the startup cache.")
    parser.add_argument("--profile-init", action="store_true", help="Print startup profile JSON and exit.")
    parser.add_argument("--profile-lexicon-only", action="store_true", help="Profile lexicon loading only and exit.")
    parser.add_argument("--profile-sample-size", type=int, default=128, help="Sample size for lexicon profile decoding.")
    parser.add_argument("--profile-skip-materialize", action="store_true", help="Skip full materialization during lexicon-only profiling.")
    parser.add_argument(
        "--profile-lightweight-materialize",
        action="store_true",
        help="Use lightweight indexed materialization during profiling when possible.",
    )
    parser.add_argument("--external-eval", action="store_true", help="Run external evaluator after each turn.")
    parser.add_argument("--external-teacher", action="store_true", help="Run external teacher after each turn.")
    parser.add_argument("--llm-order", default="settings/LLM_order.yaml", help="Path to the external LLM order YAML.")
    parser.add_argument(
        "--teacher-profiles",
        default="settings/teacher_profiles.yaml",
        help="Path to the evaluator/teacher profile YAML.",
    )
    parser.add_argument(
        "--scoring-config",
        default="settings/scoring_v1.yaml",
        help="Path to the scoring and external reward merge YAML.",
    )
    return parser



def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.profile_lexicon_only:
        print(
            json.dumps(
                profile_lexicon_load(
                    args.lexicon,
                    sample_size=args.profile_sample_size,
                    skip_materialize=args.profile_skip_materialize,
                    lightweight_materialize=args.profile_lightweight_materialize,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    engine = MinimalChatEngine(
        args.lexicon,
        runtime_dir=args.runtime_dir,
        trace_mode=args.trace_mode,
        strict_schema=not args.non_strict_schema,
        strict_relations=True,
        require_closed_relations=not args.allow_open_relations,
        startup_mode=args.startup_mode,
        startup_cache=not args.no_startup_cache,
        rebuild_startup_cache=args.rebuild_startup_cache,
        enable_external_eval=args.external_eval,
        enable_external_teacher=args.external_teacher,
        llm_order_path=args.llm_order,
        teacher_profile_path=args.teacher_profiles,
        scoring_config_path=args.scoring_config,
    )

    if args.profile_init:
        print(json.dumps(engine.startup_info, ensure_ascii=False, indent=2))
        return 0

    if args.text:
        trace = engine.run_turn(args.text)
        if args.dump_trace:
            print(json.dumps(trace, ensure_ascii=False, indent=2))
        else:
            print(trace["response"])
        return 0

    print("LSLM v4 minimal chat v1")
    print("exit / quit / Ctrl-D で終了")
    while True:
        try:
            text = input("You> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            break
        trace = engine.run_turn(text)
        print(f"AI > {trace['response']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
