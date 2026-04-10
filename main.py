from __future__ import annotations

import argparse
import sys
from typing import Sequence

SUPPORTED_MODES = {"chat", "chat_v1", "loop-learning", "loop_learning"}
DEFAULT_MODE = "chat"


MODE_HELP = {
    "chat": "Minimal relation-first interactive chat / single-turn execution.",
    "loop-learning": "Episode collection and auto-input learning runner.",
}


def _extract_mode_and_passthrough(argv: Sequence[str] | None) -> tuple[str, list[str], bool]:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    mode = DEFAULT_MODE
    passthrough: list[str] = []
    skip_next = False
    explicit_mode = False

    for index, token in enumerate(raw_args):
        if skip_next:
            skip_next = False
            continue
        if token == "--mode":
            explicit_mode = True
            if index + 1 < len(raw_args):
                mode = str(raw_args[index + 1]).strip().lower() or DEFAULT_MODE
                skip_next = True
            continue
        if token.startswith("--mode="):
            explicit_mode = True
            mode = token.split("=", 1)[1].strip().lower() or DEFAULT_MODE
            continue
        passthrough.append(token)
    return mode, passthrough, explicit_mode



def _build_global_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local Small Language Model v4 entrypoint.",
        epilog=(
            "Examples:\n"
            "  python main.py --mode chat --lexicon libs/dict.lsdx\n"
            "  python main.py --mode chat --lexicon libs/dict.lsdx --text \"猫は動物？\"\n"
            "  python main.py --mode loop-learning --lexicon libs/dict.lsdx --auto-input --max-episodes 32\n\n"
            "Mode-specific help:\n"
            "  python main.py --mode chat --help\n"
            "  python main.py --mode loop-learning --help"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", default=DEFAULT_MODE, choices=sorted(SUPPORTED_MODES), help="Execution mode.")
    return parser



def _print_unsupported_mode(mode: str) -> int:
    supported = ", ".join(sorted(SUPPORTED_MODES))
    print(
        f"Unsupported mode: {mode}\n"
        f"Supported modes in this snapshot: {supported}.\n"
        "Use --mode chat for the minimal relation-first chat pipeline, or --mode loop-learning for episode collection.",
        file=sys.stderr,
    )
    return 2



def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    mode, passthrough, explicit_mode = _extract_mode_and_passthrough(raw_args)

    if any(token in {"-h", "--help"} for token in raw_args) and not explicit_mode:
        _build_global_parser().print_help()
        return 0

    if mode not in SUPPORTED_MODES:
        return _print_unsupported_mode(mode)

    if mode in {"chat", "chat_v1"}:
        from src.apps.chat_v1 import main as chat_main

        return chat_main(passthrough)

    if mode in {"loop-learning", "loop_learning"}:
        from src.apps.loop_learning_v1 import main as loop_learning_main

        return loop_learning_main(passthrough)

    return _print_unsupported_mode(mode)


if __name__ == "__main__":
    raise SystemExit(main())
