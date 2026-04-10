from __future__ import annotations

import argparse
from typing import Iterable

TRACE_MODE_CHOICES = ["minimal", "standard", "deep_trace"]
STARTUP_MODE_CHOICES = ["auto", "fast", "full"]


def add_engine_runtime_args(
    parser: argparse.ArgumentParser,
    *,
    include_text: bool = False,
    include_dump_trace: bool = False,
    include_debug: bool = True,
    chat_help: bool = False,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--lexicon",
        default="libs/dict.lsdx",
        help="Path to the JSON/LSD/LSDX lexicon container.",
    )
    if include_text:
        parser.add_argument(
            "--text",
            help=(
                "Single-turn input text. If omitted, starts interactive mode."
                if chat_help
                else "Inline learning input. Repeat this flag to add multiple items."
            ),
        )
    parser.add_argument("--runtime-dir", default="runtime", help="Runtime directory for logs and traces.")
    parser.add_argument(
        "--trace-mode",
        default="standard",
        choices=TRACE_MODE_CHOICES,
        help="Trace verbosity mode.",
    )
    if include_debug:
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug.log output. When --trace-mode is omitted, this also promotes trace mode to deep_trace.",
        )
    parser.add_argument("--allow-open-relations", action="store_true", help="Do not require a closed relation graph.")
    parser.add_argument("--non-strict-schema", action="store_true", help="Relax top-level schema validation.")
    if include_dump_trace:
        parser.add_argument("--dump-trace", action="store_true", help="Print the full trace JSON instead of only the response.")
    parser.add_argument(
        "--startup-mode",
        default="auto",
        choices=STARTUP_MODE_CHOICES,
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
    parser.add_argument(
        "--no-episodes",
        action="store_true",
        help="Deprecated. chat mode no longer writes runtime/episodes/latest.jsonl.",
    )
    parser.add_argument("--scoring-config", help="Optional path to scoring_v1 YAML config.")
    parser.add_argument("--llm-order", default="settings/LLM_order.yaml", help="Path to the external LLM order YAML.")
    parser.add_argument(
        "--teacher-profiles",
        default="settings/teacher_profiles.yaml",
        help="Path to the evaluator/teacher profile YAML.",
    )
    return parser


def resolve_trace_mode(args: argparse.Namespace) -> str:
    if getattr(args, "trace_mode", "standard") == "standard" and getattr(args, "debug", False):
        return "deep_trace"
    return str(getattr(args, "trace_mode", "standard"))


def has_help_flag(argv: Iterable[str] | None) -> bool:
    return any(token in {"-h", "--help"} for token in (argv or []))
