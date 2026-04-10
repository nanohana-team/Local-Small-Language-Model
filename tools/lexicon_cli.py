from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import augment_conversation_lexicon
from tools import bootstrap_japanese_lexicon
from tools import convert_binary_to_dict
from tools import convert_dict_to_binary
from tools import profile_lexicon_load

Command = Callable[[Sequence[str] | None], int]


def _run_convert_to_binary(argv: Sequence[str] | None) -> int:
    return convert_dict_to_binary.main(argv)


def _run_convert_from_binary(argv: Sequence[str] | None) -> int:
    return convert_binary_to_dict.main(argv)


def _run_profile(argv: Sequence[str] | None) -> int:
    return profile_lexicon_load.main(argv)


def _run_augment_conversation(argv: Sequence[str] | None) -> int:
    return augment_conversation_lexicon.main(argv)


def _run_bootstrap_ja(argv: Sequence[str] | None) -> int:
    return bootstrap_japanese_lexicon.main(argv)


COMMANDS: dict[str, tuple[str, Command]] = {
    "convert-to-binary": ("Convert JSON/LSD/LSDX lexicons into normalized .lsd/.lsdx/.json containers.", _run_convert_to_binary),
    "convert-from-binary": ("Export normalized lexicons back to JSON-oriented views.", _run_convert_from_binary),
    "profile-load": ("Profile indexed/full lexicon startup and sample decoding.", _run_profile),
    "augment-conversation": ("Merge conversation-oriented seed lexicons into an existing base lexicon.", _run_augment_conversation),
    "bootstrap-ja": ("Bootstrap a Japanese v4 lexicon from WordNet/Sudachi/UniDic sources.", _run_bootstrap_ja),
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified lexicon maintenance CLI for Local Small Language Model v4.",
        epilog=(
            "Examples:\n"
            "  python tools/lexicon_cli.py convert-to-binary input.json --verify\n"
            "  python tools/lexicon_cli.py convert-from-binary libs/dict.lsdx --style lexical\n"
            "  python tools/lexicon_cli.py profile-load libs/dict.lsdx --sample-size 256\n"
            "  python tools/lexicon_cli.py augment-conversation --base libs/dict.lsdx --output runtime/dictionaries/dict_conversation_plus.lsdx\n"
            "  python tools/lexicon_cli.py bootstrap-ja runtime/dictionaries/bootstrapped_v1.json --force"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", nargs="?", choices=sorted(COMMANDS), help="Subcommand to run.")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed through to the selected subcommand.")
    return parser



def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    _, runner = COMMANDS[args.command]
    passthrough = list(args.args)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    return runner(passthrough)


if __name__ == "__main__":
    raise SystemExit(main())
