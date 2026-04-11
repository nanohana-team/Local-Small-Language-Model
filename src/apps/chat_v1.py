from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict

from src.core.convergence import run_convergence_v1
from src.core.divergence import analyze_input_v1, run_divergence_v1
from src.core.evaluation import apply_external_feedback, build_external_signal, summarize_external_result
from src.core.io.lsd_lexicon import (
    inspect_lexicon_storage,
    load_indexed_lsd_lexicon_container,
    load_lexicon_container,
    profile_lexicon_load,
)
from src.core.logging import TraceLogger
from src.core.planning import build_plan_v1
from src.core.relation import RelationIndex, build_relation_index, validate_relation_graph
from src.core.scoring import score_turn_v1
from src.core.slotting import fill_slots_v1
from src.core.surface import render_surface_v1
from src.llm import ExternalTeacherOrchestrator
from src.apps.cli_common import add_engine_runtime_args, resolve_trace_mode

STARTUP_CACHE_VERSION = 2


class MinimalChatEngine:
    """LSLM v4 minimal vertical slice."""

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
        write_episodes: bool = True,
        scoring_config_path: str | None = None,
        llm_order_path: str | Path = "settings/LLM_order.yaml",
        teacher_profile_path: str | Path = "settings/teacher_profiles.yaml",
        debug_enabled: bool = False,
    ) -> None:
        self.lexicon_path = Path(lexicon_path)
        self.runtime_dir = Path(runtime_dir)
        self.logger = TraceLogger(
            self.runtime_dir,
            mode=trace_mode,
            scoring_config_path=scoring_config_path,
            debug_enabled=debug_enabled,
        )
        self.startup_info: Dict[str, Any] = {}
        self.enable_external_eval = enable_external_eval
        self.enable_external_teacher = enable_external_teacher
        self.write_episodes = write_episodes
        self.scoring_config_path = scoring_config_path
        self.external_orchestrator = None
        if enable_external_eval or enable_external_teacher:
            self.external_orchestrator = ExternalTeacherOrchestrator(
                llm_order_path=llm_order_path,
                teacher_profile_path=teacher_profile_path,
                logger=self.logger,
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
        self.logger.update_session_manifest(
            startup=self.startup_info,
            lexicon={
                "path": str(self.lexicon_path),
                "concept_count": len(self.index.concepts),
                "lexical_entry_count": len(self.index.lexical_entries),
                "slot_frame_count": len(self.index.slot_frames),
            },
        )
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

    def run_turn(self, text: str, *, run_context: Dict[str, Any] | None = None, record_trace: bool = True) -> Dict[str, Any]:
        turn_id = self.logger.next_turn_id()
        started = time.perf_counter()
        run_context = dict(run_context or {})
        self.logger.debug(f"turn_started turn_id={turn_id} input={text!r}")

        input_started = time.perf_counter()
        input_analysis = analyze_input_v1(text, self.index)
        input_analysis_ms = (time.perf_counter() - input_started) * 1000.0

        plan_started = time.perf_counter()
        plan = build_plan_v1(
            text,
            topic_count=len(input_analysis.input_features.get("seed_concepts", [])),
            unknown_words=len(input_analysis.input_features.get("unknown_words", [])),
            unknown_focus=input_analysis.unknown_focus,
        )
        plan_ms = (time.perf_counter() - plan_started) * 1000.0
        self.logger.debug(
            "plan_built "
            f"turn_id={turn_id} intent={plan.intent} mode={plan.response_mode} "
            f"seeds={len(input_analysis.input_features.get('seed_concepts', []))} "
            f"unknown_words={len(input_analysis.input_features.get('unknown_words', []))}"
        )

        divergence_started = time.perf_counter()
        divergence = run_divergence_v1(text, plan, self.index, input_analysis=input_analysis)
        divergence_ms = (time.perf_counter() - divergence_started) * 1000.0
        self.logger.debug(
            "divergence_completed "
            f"turn_id={turn_id} candidates={len(divergence.candidate_concepts)} "
            f"seed_matches={len(divergence.seed_matches)} explored_relations={len(divergence.explored_relations)}"
        )

        convergence_started = time.perf_counter()
        convergence = run_convergence_v1(divergence, plan, self.index)
        convergence_ms = (time.perf_counter() - convergence_started) * 1000.0
        self.logger.debug(
            "convergence_completed "
            f"turn_id={turn_id} accepted_concepts={len(convergence.accepted_concepts)} "
            f"rejected_concepts={len(convergence.rejected_concepts)} accepted_relations={len(convergence.accepted_relations)} "
            f"rejected_relations={len(convergence.rejected_relations)}"
        )

        slot_started = time.perf_counter()
        slots = fill_slots_v1(plan, divergence, convergence, self.index)
        slot_ms = (time.perf_counter() - slot_started) * 1000.0
        self.logger.debug(
            "slot_completed "
            f"turn_id={turn_id} filled={len(slots.filled_slots)} missing={len(slots.missing_slots)}"
        )

        surface_started = time.perf_counter()
        surface = render_surface_v1(plan, slots, accepted_relations=convergence.accepted_relations)
        surface_ms = (time.perf_counter() - surface_started) * 1000.0
        self.logger.debug(
            "surface_completed "
            f"turn_id={turn_id} mode={surface['sentence_plan'].get('mode')} response={surface['final_text']!r}"
        )

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

        reward = dict(turn_score.reward)
        evaluator_summary = None
        teacher_summary = None
        evaluator_ms = 0.0

        trace = {
            "schema_version": "trace_v1",
            "record_type": "trace",
            "session_id": self.logger.session_id,
            "turn_id": turn_id,
            "input": text,
            "input_features": divergence.input_features,
            "plan": plan.to_dict(),
            "divergence_candidates": [candidate.to_dict() for candidate in divergence.candidate_concepts[:12]],
            "seed_matches": [match.to_dict() for match in divergence.seed_matches[:12]],
            "explored_relations": divergence.explored_relations,
            "convergence_candidates": convergence.accepted_concepts,
            "rejected_candidates": convergence.rejected_concepts,
            "accepted_relations": convergence.accepted_relations,
            "rejected_relations": convergence.rejected_relations,
            "filled_slots": slots.filled_slots,
            "missing_slots": slots.missing_slots,
            "slot_evidence": slots.slot_evidence,
            "surface_plan": surface["sentence_plan"],
            "response": surface["final_text"],
            "scores": turn_score.scores,
            "reward": reward,
            "feedback": turn_score.feedback,
            "scoring_details": turn_score.details,
            "timing": {
                "total_ms": round(total_ms, 3),
                "input_analysis_ms": round(input_analysis_ms, 3),
                "plan_ms": round(plan_ms, 3),
                "divergence_ms": round(divergence_ms, 3),
                "convergence_ms": round(convergence_ms, 3),
                "slot_ms": round(slot_ms, 3),
                "surface_ms": round(surface_ms, 3),
                "evaluator_ms": 0.0,
            },
            "run_context": run_context or None,
            "trace_mode": self.logger.mode,
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
            }
            if self.enable_external_eval:
                external_started = time.perf_counter()
                evaluator_result = self.external_orchestrator.evaluate_turn(external_payload)
                evaluator_ms = (time.perf_counter() - external_started) * 1000.0
                evaluator_summary = summarize_external_result(evaluator_result)
                trace["external_evaluator"] = evaluator_summary
                trace["timing"]["evaluator_ms"] = round(evaluator_ms, 3)

            if self.enable_external_teacher:
                teacher_result = self.external_orchestrator.teach_turn(external_payload)
                teacher_summary = summarize_external_result(teacher_result)
                trace["external_teacher"] = teacher_summary

            if evaluator_summary is not None or teacher_summary is not None:
                reward = apply_external_feedback(
                    reward,
                    evaluator_summary,
                    teacher_summary,
                    scoring_config_path=self.scoring_config_path,
                )
                trace["reward"] = reward
                trace["external_signal"] = build_external_signal(
                    evaluator_summary,
                    teacher_summary,
                    scoring_config_path=self.scoring_config_path,
                )

        self.logger.debug(
            f"turn_completed turn_id={turn_id} accepted_concepts={len(convergence.accepted_concepts)} total_ms={total_ms:.3f} decision={trace['scoring_details'].get('decision')}"
        )
        if record_trace:
            self.logger.record_trace(trace)
        return trace


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LSLM v4 minimal relation-first chat pipeline.")
    add_engine_runtime_args(
        parser,
        include_text=True,
        include_dump_trace=True,
        include_debug=True,
        chat_help=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

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

    trace_mode = resolve_trace_mode(args)

    engine = MinimalChatEngine(
        args.lexicon,
        runtime_dir=args.runtime_dir,
        trace_mode=trace_mode,
        strict_schema=not args.non_strict_schema,
        strict_relations=True,
        require_closed_relations=not args.allow_open_relations,
        startup_mode=args.startup_mode,
        startup_cache=not args.no_startup_cache,
        rebuild_startup_cache=args.rebuild_startup_cache,
        enable_external_eval=args.external_eval,
        enable_external_teacher=args.external_teacher,
        write_episodes=not args.no_episodes,
        scoring_config_path=args.scoring_config,
        llm_order_path=args.llm_order,
        teacher_profile_path=args.teacher_profiles,
        debug_enabled=args.debug,
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

    print("LSLM v4 minimal chat")
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
