from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.apps.chat_v1 import MinimalChatEngine
from src.apps.cli_common import add_engine_runtime_args, resolve_trace_mode
from src.core.io.lsd_lexicon import profile_lexicon_load
from src.core.records import AdditionalLexiconStore, EpisodeWriter, build_episode_v1
from src.core.records.improvement_candidates import build_teacher_improvement_candidate
from src.core.records.improvement_writer import ImprovementCandidateWriter
from src.llm.teacher_adapter import ExternalTeacherOrchestrator

try:
    JST = ZoneInfo("Asia/Tokyo")
except ZoneInfoNotFoundError:
    JST = timezone(timedelta(hours=9))

DEFAULT_AUTO_INPUTS: list[str] = [
    "おはよう",
    "こんにちは",
    "ありがとう",
    "LSLM v4って何？",
    "犬と猫の違いは？",
    "空はなぜ青いの？",
    "勉強に集中するコツは？",
    "眠れないときはどうすればいい？",
    "パソコンが重いときの確認手順は？",
    "雨の日に気分を上げる方法は？",
    "AIと辞書の違いを説明して",
    "relationって何？",
    "比較するときに大事なことは？",
    "日本語を自然にするには？",
    "短く自己紹介して",
    "猫は動物？",
]




class LoopLearningError(RuntimeError):
    pass


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LSLM v4 loop-learning episode runner.")
    add_engine_runtime_args(parser, include_debug=True)
    parser.add_argument("--dataset", help="Learning input dataset. Supports .txt, .json, and .jsonl.")
    parser.add_argument("--text", action="append", help="Inline learning input. Repeat this flag to add multiple items.")
    parser.add_argument("--auto-input", action="store_true", help="Generate learning inputs automatically. LLM generation is tried first, then falls back to the built-in prompt bank.")
    parser.add_argument("--auto-input-count", type=int, default=0, help="Requested number of auto-generated inputs. Defaults to max_episodes, capped to a safe range.")
    parser.add_argument("--auto-input-topic", action="append", help="Optional topic hint for auto input generation. Repeat to add multiple topics.")
    parser.add_argument("--auto-input-no-llm", action="store_true", help="Disable LLM-based input generation and use only the built-in prompt bank.")
    parser.add_argument("--max-episodes", type=int, default=32, help="Maximum number of episodes to run.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the dataset each cycle.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed used for shuffle order.")
    parser.add_argument("--stop-on-error", action="store_true", help="Abort the loop on the first episode failure.")
    parser.add_argument("--progress-every", type=int, default=1, help="Print progress every N completed episodes.")
    parser.add_argument(
        "--write-improvement-candidates",
        action="store_true",
        help="Extract teacher improvement candidates into runtime/review_candidates/latest.jsonl.",
    )
    parser.add_argument(
        "--dump-summary",
        action="store_true",
        help="Print the final summary JSON at the end of the run.",
    )
    parser.add_argument(
        "--no-unknown-word-llm",
        action="store_true",
        help="Disable LLM-based unknown word enrichment during loop-learning.",
    )
    parser.add_argument(
        "--unknown-lexicon-path",
        default="runtime/dictionaries/additional_dict.lsdx",
        help="Path to the auxiliary lexicon written from loop-learning unknown words.",
    )
    parser.add_argument(
        "--unknown-word-batch-size",
        type=int,
        default=4,
        help="Maximum number of new unknown terms to query per episode.",
    )
    return parser


class LoopLearningRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.runtime_dir = Path(args.runtime_dir)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.engine = MinimalChatEngine(
            args.lexicon,
            runtime_dir=args.runtime_dir,
            trace_mode=resolve_trace_mode(args),
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
            debug_enabled=bool(getattr(args, "debug", False)),
        )
        self.episode_writer = EpisodeWriter(args.runtime_dir) if not args.no_episodes else None
        self.candidate_writer = (
            ImprovementCandidateWriter(args.runtime_dir)
            if args.write_improvement_candidates and args.external_teacher
            else None
        )
        self._auto_input_meta: dict[str, Any] = {"source": "disabled", "requested_count": 0}
        self.additional_lexicon = AdditionalLexiconStore(args.unknown_lexicon_path)
        self.additional_lexicon.merge_into_engine(self.engine)
        self.unknown_word_orchestrator = None
        if not args.no_unknown_word_llm:
            self.unknown_word_orchestrator = ExternalTeacherOrchestrator(
                llm_order_path=args.llm_order,
                teacher_profile_path=args.teacher_profiles,
                logger=self.engine.logger,
            )
        self.unknown_word_stats: dict[str, Any] = {
            "enabled": not args.no_unknown_word_llm,
            "queries": 0,
            "query_failures": 0,
            "requested_terms": 0,
            "added_terms": 0,
            "skipped_terms": 0,
            "seen_terms": [],
            "added_surface_samples": [],
            "path": str(Path(args.unknown_lexicon_path)),
        }
        self.engine.logger.update_session_manifest(
            loop_learning={
                "write_episodes": self.episode_writer is not None,
                "write_improvement_candidates": self.candidate_writer is not None,
                "unknown_word_enrichment_enabled": self.unknown_word_orchestrator is not None,
                "unknown_lexicon_path": str(self.additional_lexicon.path),
            }
        )

    def _input_source_label(self) -> str:
        if self.args.auto_input:
            return "auto_input"
        if self.args.dataset:
            return "dataset"
        return "inline_text"

    def _empty_unknown_enrichment_status(self, trace: Mapping[str, Any], status: str, reason: str) -> dict[str, Any]:
        input_features = trace.get("input_features") if isinstance(trace.get("input_features"), Mapping) else {}
        raw_terms = input_features.get("unknown_term_candidates")
        if not isinstance(raw_terms, list) or not raw_terms:
            raw_terms = input_features.get("unknown_words") if isinstance(input_features.get("unknown_words"), list) else []
        return {
            "status": status,
            "reason": reason,
            "unknown_words": [str(term) for term in raw_terms if term is not None][:8],
            "requested_terms": [],
            "enriched_words": [],
            "skipped_terms": [],
            "overlay_path": str(self.additional_lexicon.path),
        }

    def _maybe_enrich_unknown_words(self, trace: Mapping[str, Any]) -> dict[str, Any]:
        if self.unknown_word_orchestrator is None:
            return self._empty_unknown_enrichment_status(trace, "disabled", "unknown_word_llm_disabled")
        input_features = trace.get("input_features") if isinstance(trace.get("input_features"), Mapping) else {}
        raw_terms = input_features.get("unknown_term_candidates")
        if not isinstance(raw_terms, list) or not raw_terms:
            raw_terms = input_features.get("unknown_words") if isinstance(input_features.get("unknown_words"), list) else []
        unseen_terms = self.additional_lexicon.unseen_terms(raw_terms, base_index=self.engine.index)
        batch_size = max(1, min(int(self.args.unknown_word_batch_size or 4), 8))
        requested_terms = unseen_terms[:batch_size]
        if not raw_terms:
            return self._empty_unknown_enrichment_status(trace, "not_needed", "no_unknown_terms")
        if not requested_terms:
            return self._empty_unknown_enrichment_status(trace, "not_needed", "already_known_or_recorded")

        self.unknown_word_stats["queries"] += 1
        self.unknown_word_stats["requested_terms"] += len(requested_terms)
        self.unknown_word_stats["seen_terms"] = _dedupe_preserve_order(
            list(self.unknown_word_stats.get("seen_terms", [])) + requested_terms
        )[:64]

        payload = {
            "requested_terms": requested_terms,
            "user_input": str(trace.get("input") or ""),
            "plan": trace.get("plan") if isinstance(trace.get("plan"), Mapping) else {},
        }
        result = self.unknown_word_orchestrator.run_profile("lexicon_enricher", payload)
        if result.error:
            self.unknown_word_stats["query_failures"] += 1
        update = self.additional_lexicon.apply_llm_entries(
            requested_terms=requested_terms,
            parsed_payload=result.parsed,
            provider=result.provider,
            model=result.model,
            prompt_version=result.prompt_version,
            raw_text=result.raw_text,
            error=result.error,
        )
        if update.added_terms:
            merged_count = self.additional_lexicon.merge_into_engine(self.engine)
            self.engine.logger.info(
                "unknown_word_overlay_updated "
                f"requested={len(requested_terms)} added={len(update.added_terms)} merged={merged_count} path={self.additional_lexicon.path}"
            )
        elif update.error:
            self.engine.logger.warning(
                f"unknown_word_overlay_failed requested={requested_terms} error={update.error}"
            )

        self.unknown_word_stats["added_terms"] += len(update.added_terms)
        self.unknown_word_stats["skipped_terms"] += len(update.skipped_terms)
        self.unknown_word_stats["added_surface_samples"] = _dedupe_preserve_order(
            list(self.unknown_word_stats.get("added_surface_samples", [])) + list(update.added_terms)
        )[:64]
        status = "query_failed" if update.error else ("enriched" if update.added_terms else "no_new_entries")
        return {
            "status": status,
            "reason": update.error or ("overlay_updated" if update.added_terms else "llm_returned_no_new_entries"),
            "unknown_words": [str(term) for term in raw_terms if term is not None][:8],
            "requested_terms": list(update.requested_terms),
            "enriched_words": list(update.added_terms),
            "skipped_terms": list(update.skipped_terms),
            "overlay_path": str(self.additional_lexicon.path),
            "provider": update.provider,
            "model": update.model,
            "error": update.error,
        }

    def run(self) -> dict[str, Any]:
        prompts = _load_learning_inputs(self.args, auto_input_meta=self._auto_input_meta, logger=self.engine.logger)
        if not prompts:
            raise LoopLearningError("loop-learning requires at least one input. Use --dataset, --text, or --auto-input.")
        max_episodes = max(1, int(self.args.max_episodes))
        rng = random.Random(int(self.args.seed))
        started_at = datetime.now(JST)
        self.engine.logger.update_session_manifest(
            loop_learning={
                "input_source": self._input_source_label(),
                "auto_input": dict(self._auto_input_meta),
                "max_episodes": max_episodes,
                "external_eval": bool(self.args.external_eval),
                "external_teacher": bool(self.args.external_teacher),
            }
        )
        if self.args.auto_input:
            print(
                "[LOOP] auto_input "
                f"source={self._auto_input_meta.get('source')} "
                f"used_llm={bool(self._auto_input_meta.get('used_llm', False))} "
                f"provider={self._auto_input_meta.get('provider') or 'none'} "
                f"model={self._auto_input_meta.get('model') or 'none'} "
                f"generated={self._auto_input_meta.get('generated_count')} "
                f"error={self._auto_input_meta.get('error') or 'none'}"
            )

        decision_counter: Counter[str] = Counter()
        intent_counter: Counter[str] = Counter()
        failures: list[dict[str, Any]] = []
        reward_totals: list[float] = []
        external_totals: list[float] = []
        completed = 0

        prompt_iter = _cycling_prompts(prompts, max_episodes=max_episodes, shuffle=self.args.shuffle, rng=rng)
        for episode_index, prompt in enumerate(prompt_iter, start=1):
            try:
                trace = self.engine.run_turn(
                    prompt,
                    run_context={
                        "runner": "loop_learning",
                        "episode_index": episode_index,
                        "input_source": self._input_source_label(),
                    },
                    record_trace=False,
                )
                completed += 1
            except Exception as exc:
                failure = {
                    "episode_index": episode_index,
                    "input": prompt,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                failures.append(failure)
                print(f"[LOOP] episode {episode_index}/{max_episodes} failed: {failure['error']}")
                if self.args.stop_on_error:
                    break
                continue

            decision = str((trace.get("scoring_details") or {}).get("decision") or "review")
            decision_counter[decision] += 1
            plan = trace.get("plan") if isinstance(trace.get("plan"), Mapping) else {}
            intent_counter[str(plan.get("intent") or "unknown")] += 1

            total_reward = (trace.get("reward") or {}).get("total")
            try:
                reward_totals.append(float(total_reward))
            except (TypeError, ValueError):
                pass

            external_reward = (trace.get("reward") or {}).get("external")
            try:
                if external_reward is not None:
                    external_totals.append(float(external_reward))
            except (TypeError, ValueError):
                pass

            unknown_overlay_update = self._maybe_enrich_unknown_words(trace)
            trace["unknown_word_enrichment"] = unknown_overlay_update
            self.engine.logger.record_trace(trace)

            if self.episode_writer is not None:
                episode = build_episode_v1(
                    trace,
                    episode_index=episode_index,
                    input_source=self._input_source_label(),
                    unknown_word_enrichment=unknown_overlay_update,
                    auto_input_meta=self._auto_input_meta,
                    trace_runtime_path=str(self.engine.logger.latest_trace_path),
                )
                self.episode_writer.write(episode)

            if self.candidate_writer is not None:
                candidate = build_teacher_improvement_candidate(
                    session_id=str(trace.get("session_id") or self.engine.logger.session_id),
                    turn_id=str(trace.get("turn_id") or f"turn_{episode_index:04d}"),
                    user_input=str(trace.get("input") or prompt),
                    response=str(trace.get("response") or ""),
                    plan=plan,
                    teacher_summary=trace.get("external_teacher") if isinstance(trace.get("external_teacher"), Mapping) else None,
                    external_signal=trace.get("external_signal") if isinstance(trace.get("external_signal"), Mapping) else None,
                )
                if candidate is not None:
                    self.candidate_writer.write(candidate)

            if self.args.progress_every > 0 and (completed % self.args.progress_every == 0 or completed == max_episodes):
                total_display = (trace.get("reward") or {}).get("total")
                unknown_note = ""
                if isinstance(unknown_overlay_update, Mapping) and unknown_overlay_update.get("enriched_words"):
                    unknown_note = f" unknown_added={','.join(unknown_overlay_update.get('enriched_words') or [])}"
                print(
                    f"[LOOP] episode {completed}/{max_episodes} intent={plan.get('intent')} "
                    f"decision={decision} reward_total={total_display} input={prompt}{unknown_note}"
                )

            if completed >= max_episodes:
                break

        finished_at = datetime.now(JST)
        summary = {
            "schema_version": "loop_learning_summary_v1",
            "session_id": self.engine.logger.session_id,
            "started_at_jst": started_at.isoformat(),
            "finished_at_jst": finished_at.isoformat(),
            "runtime_dir": str(self.runtime_dir),
            "lexicon": str(Path(self.args.lexicon)),
            "trace_mode": self.args.trace_mode,
            "episodes_requested": max_episodes,
            "episodes_completed": completed,
            "episodes_failed": len(failures),
            "decision_counts": dict(decision_counter),
            "intent_counts": dict(intent_counter),
            "reward_total_avg": _mean_or_none(reward_totals),
            "reward_external_avg": _mean_or_none(external_totals),
            "write_improvement_candidates": bool(self.candidate_writer is not None),
            "external_eval": bool(self.args.external_eval),
            "external_teacher": bool(self.args.external_teacher),
            "auto_input": dict(self._auto_input_meta),
            "unknown_word_enrichment": {
                "enabled": bool(self.unknown_word_orchestrator is not None),
                "additional_lexicon_path": str(self.additional_lexicon.path),
                "queries": int(self.unknown_word_stats.get("queries", 0)),
                "query_failures": int(self.unknown_word_stats.get("query_failures", 0)),
                "requested_terms": int(self.unknown_word_stats.get("requested_terms", 0)),
                "added_terms": int(self.unknown_word_stats.get("added_terms", 0)),
                "skipped_terms": int(self.unknown_word_stats.get("skipped_terms", 0)),
                "seen_term_samples": list(self.unknown_word_stats.get("seen_terms", []))[:16],
                "added_surface_samples": list(self.unknown_word_stats.get("added_surface_samples", []))[:16],
            },
            "failures": failures[:16],
        }
        _write_run_summary(self.runtime_dir, summary)
        return summary


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)



def _cycling_prompts(
    prompts: list[str],
    *,
    max_episodes: int,
    shuffle: bool,
    rng: random.Random,
) -> Iterator[str]:
    if not prompts:
        return
    emitted = 0
    working = list(prompts)
    while emitted < max_episodes:
        if shuffle and len(working) > 1:
            rng.shuffle(working)
        for prompt in working:
            yield prompt
            emitted += 1
            if emitted >= max_episodes:
                break



def _load_learning_inputs(
    args: argparse.Namespace,
    *,
    auto_input_meta: dict[str, Any] | None = None,
    logger: Any | None = None,
) -> list[str]:
    inputs: list[str] = []
    for item in args.text or []:
        normalized = _normalize_prompt(item)
        if normalized:
            inputs.append(normalized)

    if args.dataset:
        inputs.extend(_load_dataset_file(Path(args.dataset)))

    if not inputs and args.auto_input:
        generated = _build_auto_inputs(args, auto_input_meta=auto_input_meta, logger=logger)
        inputs.extend(generated)

    if auto_input_meta is not None and not args.auto_input:
        auto_input_meta.update({"source": "disabled", "requested_count": 0, "generated_count": 0})

    return _dedupe_preserve_order(inputs)



def _build_auto_inputs(
    args: argparse.Namespace,
    *,
    auto_input_meta: dict[str, Any] | None = None,
    logger: Any | None = None,
) -> list[str]:
    requested_count = int(args.auto_input_count or args.max_episodes or len(DEFAULT_AUTO_INPUTS))
    requested_count = max(8, min(requested_count, 64))
    topic_hints = _dedupe_preserve_order([_normalize_prompt(item) for item in (args.auto_input_topic or []) if _normalize_prompt(item)])

    if auto_input_meta is not None:
        auto_input_meta.update(
            {
                "source": "fallback",
                "requested_count": requested_count,
                "topic_hints": topic_hints,
                "generated_count": 0,
            }
        )

    if not args.auto_input_no_llm:
        generated, llm_meta = _generate_auto_inputs_with_llm(
            requested_count=requested_count,
            llm_order_path=args.llm_order,
            teacher_profile_path=args.teacher_profiles,
            seed=int(args.seed),
            topic_hints=topic_hints,
            logger=logger,
        )
        if auto_input_meta is not None:
            auto_input_meta.update(llm_meta)
        combined = _dedupe_preserve_order(generated + DEFAULT_AUTO_INPUTS)
        if generated:
            return combined
        if logger is not None and hasattr(logger, "warning"):
            logger.warning(
                "auto_input_fallback_selected "
                f"reason={llm_meta.get('error') or 'llm_returned_no_inputs'} requested={requested_count}"
            )
    elif logger is not None and hasattr(logger, "info"):
        logger.info("auto_input_llm_disabled requested=%s" % requested_count)

    fallback = _dedupe_preserve_order(DEFAULT_AUTO_INPUTS)
    if auto_input_meta is not None:
        auto_input_meta.update(
            {
                "source": "fallback",
                "requested_count": requested_count,
                "generated_count": len(fallback),
                "used_llm": False,
            }
        )
    return fallback



def _generate_auto_inputs_with_llm(
    *,
    requested_count: int,
    llm_order_path: str,
    teacher_profile_path: str,
    seed: int,
    topic_hints: list[str],
    logger: Any | None = None,
) -> tuple[list[str], dict[str, Any]]:
    orchestrator = ExternalTeacherOrchestrator(
        llm_order_path=llm_order_path,
        teacher_profile_path=teacher_profile_path,
        logger=logger,
    )
    profile = orchestrator.get_profile("input_generator")
    runtime_options = profile.runtime_options if profile is not None else {}
    default_topic_hints = _normalized_list(runtime_options.get("default_topic_hints"))
    example_bank = _normalized_list(runtime_options.get("example_bank"))
    resolved_topic_hints = topic_hints or default_topic_hints
    resolved_examples = example_bank[:12] or list(DEFAULT_AUTO_INPUTS[:12])
    if logger is not None and hasattr(logger, "info"):
        logger.info(
            "auto_input_llm_requested "
            f"requested={requested_count} topics={len(resolved_topic_hints)} examples={len(resolved_examples)} "
            f"profile=input_generator prompt_version={profile.prompt_version if profile is not None else 'missing'}"
        )
    payload = {
        "requested_count": requested_count,
        "seed": seed,
        "topic_hints": resolved_topic_hints,
        "examples": resolved_examples,
    }
    result = orchestrator.generate_inputs(payload)
    prompts = _extract_generated_inputs(result.parsed, requested_count=requested_count)
    meta = {
        "source": "llm" if prompts else "fallback",
        "requested_count": requested_count,
        "generated_count": len(prompts),
        "used_llm": True,
        "provider": result.provider,
        "model": result.model,
        "error": result.error,
        "latency_ms": result.latency_ms,
        "prompt_version": result.prompt_version,
        "topic_hints": resolved_topic_hints,
        "llm_attempted": True,
        "profile_mode": "input_generator",
        "profile_path": str(teacher_profile_path),
    }
    if logger is not None and hasattr(logger, "info"):
        logger.info(
            "auto_input_llm_result "
            f"source={meta['source']} generated={len(prompts)} provider={result.provider} model={result.model} "
            f"prompt_version={result.prompt_version} error={result.error or 'none'}"
        )
    return prompts, meta



def _extract_generated_inputs(parsed: Mapping[str, Any] | None, *, requested_count: int) -> list[str]:
    if not isinstance(parsed, Mapping):
        return []
    raw_candidates: list[Any] = []
    for key in ("inputs", "prompts", "items", "questions"):
        value = parsed.get(key)
        if isinstance(value, list):
            raw_candidates.extend(value)
            break
    normalized = [_normalize_prompt(item) for item in raw_candidates]
    outputs = [item for item in normalized if item]
    outputs = _dedupe_preserve_order(outputs)
    return outputs[:requested_count]


def _normalized_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized = [_normalize_prompt(item) for item in value]
    return [item for item in normalized if item]



def _load_dataset_file(path: Path) -> list[str]:
    if not path.exists():
        raise LoopLearningError(f"dataset not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _load_jsonl_prompts(path)
    if suffix == ".json":
        return _load_json_prompts(path)
    return _load_text_prompts(path)



def _load_jsonl_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LoopLearningError(f"invalid jsonl at {path}:{line_no}: {exc}") from exc
            prompt = _prompt_from_json_value(parsed)
            if prompt:
                prompts.append(prompt)
    return prompts



def _load_json_prompts(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        parsed = json.load(handle)
    if isinstance(parsed, list):
        prompts = [_prompt_from_json_value(item) for item in parsed]
        return [prompt for prompt in prompts if prompt]
    prompt = _prompt_from_json_value(parsed)
    return [prompt] if prompt else []



def _load_text_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            prompt = _normalize_prompt(raw_line)
            if prompt:
                prompts.append(prompt)
    return prompts



def _prompt_from_json_value(value: Any) -> str | None:
    if isinstance(value, str):
        return _normalize_prompt(value)
    if isinstance(value, Mapping):
        for key in ("input", "text", "prompt", "message", "user_input"):
            if key in value:
                return _normalize_prompt(value.get(key))
    return None



def _normalize_prompt(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None



def _dedupe_preserve_order(items: Iterable[str | None]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered



def _write_run_summary(runtime_dir: Path, summary: Mapping[str, Any]) -> None:
    target_dir = runtime_dir / "learning_runs"
    target_dir.mkdir(parents=True, exist_ok=True)
    latest_path = target_dir / "latest.json"
    if latest_path.exists() and latest_path.stat().st_size > 0:
        stamp = datetime.now(JST).strftime("%Y%m%d%H%M%S")
        archive_path = target_dir / f"{stamp}.json"
        counter = 1
        while archive_path.exists():
            archive_path = target_dir / f"{stamp}_{counter}.json"
            counter += 1
        latest_path.replace(archive_path)
    latest_path.write_text(json.dumps(dict(summary), ensure_ascii=False, indent=2), encoding="utf-8")



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

    runner = LoopLearningRunner(args)
    if args.profile_init:
        print(json.dumps(runner.engine.startup_info, ensure_ascii=False, indent=2))
        return 0

    summary = runner.run()
    if args.dump_summary:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(
            "[LOOP] completed "
            f"episodes={summary['episodes_completed']}/{summary['episodes_requested']} "
            f"avg_reward={summary['reward_total_avg']} failures={summary['episodes_failed']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
