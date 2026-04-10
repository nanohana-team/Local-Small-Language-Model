from __future__ import annotations

import argparse
import json

from src.core.io.lsd_lexicon import profile_lexicon_load


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile lexicon loading for JSON/LSD/LSDX containers.")
    parser.add_argument("path", help="Path to the lexicon file.")
    parser.add_argument("--sample-size", type=int, default=128, help="Number of indexed entries to sample-decode.")
    parser.add_argument("--skip-materialize", action="store_true", help="Skip full materialization and only profile indexed open + sample decode.")
    parser.add_argument(
        "--lightweight-materialize",
        action="store_true",
        help="Use lightweight indexed materialization when possible instead of full normalization.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = profile_lexicon_load(
        args.path,
        sample_size=args.sample_size,
        skip_materialize=args.skip_materialize,
        lightweight_materialize=args.lightweight_materialize,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
