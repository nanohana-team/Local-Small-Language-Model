from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from src.apps.chat_v1 import MinimalChatEngine
from src.core.io.lsd_lexicon import profile_lexicon_load

SUPPORTED_MODES = {"chat", "chat_v1"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LSLM v4 entrypoint. This reduced v4 snapshot currently exposes the minimal relation-first chat pipeline through main.py."
    )
    parser.add_argument(
        "--mode",
        default="chat",
        help="Execution mode. Supported in this snapshot: chat, chat_v1.",
    )
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
    parser.add_argument("--scoring-config", help="Optional path to scoring_v1 YAML config.")
    parser.add_argument("--profile-init", action="store_true", help="Print startup profile JSON and exit.")
    parser.add_argument("--profile-lexicon-only", action="store_true", help="Profile lexicon loading only and exit.")
    parser.add_argument("--profile-sample-size", type=int, default=128, help="Sample size for lexicon profile decoding.")
    parser.add_argument("--profile-skip-materialize", action="store_true", help="Skip full materialization during lexicon-only profiling.")
    parser.add_argument(
        "--profile-lightweight-materialize",
        action="store_true",
        help="Use lightweight indexed materialization during profiling when possible.",
    )
    return parser


def _print_unsupported_mode(mode: str) -> int:
    supported = ", ".join(sorted(SUPPORTED_MODES))
    print(
        f"Unsupported mode: {mode}\n"
        f"This reduced LSLM v4 snapshot currently supports: {supported}.\n"
        "Use --mode chat for the minimal relation-first chat pipeline.",
        file=sys.stderr,
    )
    return 2


def run_chat_mode(args: argparse.Namespace) -> int:
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

    print("LSLM v4 main.py chat mode")
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    mode = str(args.mode).strip().lower()
    if mode not in SUPPORTED_MODES:
        return _print_unsupported_mode(mode)
    return run_chat_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
