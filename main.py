from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

from src.apps.run_minimal_chat import (
    SurfaceNormalizer,
    parse_args as parse_minimal_args,
    run_pipeline,
)
from src.core.io.lsd_lexicon import load_lexicon_container
from src.core.schema import LexiconContainer
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["chat", "learn"], default="chat")
    parser.add_argument("--lexicon", default="libs/dict.lsdx")
    parser.add_argument("--trace-dir", default="runtime/traces")
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--text", default="")
    parser.add_argument("--words", nargs="*", default=None)

    # learning mode options
    parser.add_argument("--dataset-dir", default="runtime/datasets")
    parser.add_argument("--no-dataset", action="store_true")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--external-mode", choices=["heuristic", "none"], default="heuristic")
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--external-fallback", choices=["neutral", "internal"], default="neutral")
    parser.add_argument("--neutral-external-score", type=float, default=0.5)

    return parser



def build_pipeline_args(
    *,
    lexicon: str,
    trace_dir: str,
    no_trace: bool,
    debug: bool,
    text: str,
    words: Optional[Sequence[str]],
):
    argv: list[str] = [
        "--lexicon",
        lexicon,
        "--trace-dir",
        trace_dir,
    ]

    if no_trace:
        argv.append("--no-trace")

    if debug:
        argv.append("--console-debug")

    if words:
        argv.append("--words")
        argv.extend(words)
    else:
        argv.extend(["--text", text])

    return parse_minimal_args(argv)



def build_learning_args(args: argparse.Namespace) -> list[str]:
    argv: list[str] = [
        "--lexicon",
        args.lexicon,
        "--trace-dir",
        args.trace_dir,
        "--dataset-dir",
        args.dataset_dir,
        "--episodes",
        str(max(1, int(args.episodes))),
        "--external-mode",
        args.external_mode,
        "--alpha",
        str(args.alpha),
        "--beta",
        str(args.beta),
        "--external-fallback",
        args.external_fallback,
        "--neutral-external-score",
        str(args.neutral_external_score),
    ]

    if args.no_trace:
        argv.append("--no-trace")

    if args.no_dataset:
        argv.append("--no-dataset")

    if args.debug:
        argv.append("--debug")

    if args.words:
        argv.append("--words")
        argv.extend(args.words)
    elif args.text:
        argv.extend(["--text", args.text])

    return argv



def dispatch_learning_mode(args: argparse.Namespace) -> int:
    from src.apps.chat_learning import main as chat_learning_main

    learning_argv = build_learning_args(args)
    return int(chat_learning_main(learning_argv))



def run_chat_mode(args: argparse.Namespace) -> int:
    setup_logging(
        app_name="lslm",
        console_level=logging.DEBUG if args.debug else logging.INFO,
    )

    lexicon_path = Path(args.lexicon)
    LOGGER.info("lexicon.load.start path=%s", lexicon_path)

    container_raw = load_lexicon_container(lexicon_path)
    lexicon = LexiconContainer.from_dict(container_raw)

    LOGGER.info("lexicon.load.done entries=%s", len(lexicon.entries))

    normalizer = SurfaceNormalizer(lexicon)

    if args.text or args.words:
        pipeline_args = build_pipeline_args(
            lexicon=args.lexicon,
            trace_dir=args.trace_dir,
            no_trace=args.no_trace,
            debug=args.debug,
            text=args.text,
            words=args.words,
        )
        response_text, _ = run_pipeline(pipeline_args, lexicon, normalizer)
        print(response_text)
        return 0

    print("LSLM v3 interactive mode")

    while True:
        try:
            raw_text = input(">>> ").strip()
        except KeyboardInterrupt:
            print()
            return 0

        if not raw_text:
            continue

        if raw_text.lower() in ("exit", "quit"):
            return 0

        try:
            pipeline_args = build_pipeline_args(
                lexicon=args.lexicon,
                trace_dir=args.trace_dir,
                no_trace=args.no_trace,
                debug=args.debug,
                text=raw_text,
                words=None,
            )

            response_text, _ = run_pipeline(
                pipeline_args,
                lexicon,
                normalizer,
            )

            print(response_text)

        except Exception:
            LOGGER.exception("interactive_turn_failed")
            print("[ERROR] 応答生成に失敗しました")



def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mode == "learn":
        return dispatch_learning_mode(args)

    return run_chat_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
