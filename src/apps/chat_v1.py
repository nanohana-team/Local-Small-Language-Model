from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

from src.core.convergence import run_convergence_v1
from src.core.divergence import run_divergence_v1
from src.core.io.lsd_lexicon import load_lexicon_container
from src.core.logging import TraceLogger
from src.core.planning import build_plan_v1
from src.core.relation import build_relation_index, validate_relation_graph
from src.core.slotting import fill_slots_v1
from src.core.surface import render_surface_v1


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
    ) -> None:
        self.lexicon_path = Path(lexicon_path)
        self.runtime_dir = Path(runtime_dir)
        self.logger = TraceLogger(self.runtime_dir, mode=trace_mode)
        self.container = load_lexicon_container(self.lexicon_path)
        self.index = build_relation_index(self.container)
        self.validation_report = validate_relation_graph(
            self.container,
            strict_schema=strict_schema,
            strict_relations=strict_relations,
            require_closed_relations=require_closed_relations,
            prebuilt_index=self.index,
        )
        if not self.validation_report.get("ok", False):
            preview = "\n".join(self.validation_report.get("errors", [])[:20])
            raise ValueError(f"Relation validation failed:\n{preview}")
        self.logger.info(
            f"minimal_chat_engine_ready concepts={len(self.index.concepts)} lexical_entries={len(self.index.lexical_entries)}"
        )
        for warning in self.validation_report.get("warnings", [])[:20]:
            self.logger.warning(f"relation_validation_warning {warning}")

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

        scores = {
            "plan_fitness": 1.0 if not plan.needs_clarification else 0.45,
            "relation_coverage": min(1.0, len(convergence.accepted_relations) / 4.0),
            "divergence_relevance": min(1.0, len(divergence.candidate_concepts) / 6.0),
            "convergence_fitness": min(1.0, len(convergence.accepted_concepts) / 4.0),
            "slot_fitness": 0.0 if plan.required_slots else 1.0,
            "input_retention": 1.0 if divergence.seed_matches else 0.25,
            "dangling_rate": 0.0,
        }
        if plan.required_slots:
            scores["slot_fitness"] = max(0.0, 1.0 - len(slots.missing_slots) / len(plan.required_slots))

        reward_internal = round(
            0.20 * scores["plan_fitness"]
            + 0.20 * scores["relation_coverage"]
            + 0.15 * scores["divergence_relevance"]
            + 0.20 * scores["convergence_fitness"]
            + 0.15 * scores["slot_fitness"]
            + 0.10 * scores["input_retention"],
            6,
        )
        reward = {"internal": reward_internal, "external": None, "total": reward_internal}

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
        }
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
    return parser



def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    engine = MinimalChatEngine(
        args.lexicon,
        runtime_dir=args.runtime_dir,
        trace_mode=args.trace_mode,
        strict_schema=not args.non_strict_schema,
        strict_relations=True,
        require_closed_relations=not args.allow_open_relations,
    )

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
