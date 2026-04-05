from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import List

from src.apps.chat_learning_central import (
    ChatLearningCentral,
    build_chat_learning_central,
)


def load_dotenv_file(dotenv_path: str | Path = ".env", override: bool = False) -> None:
    path = Path(dotenv_path)
    if not path.exists():
        return

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8-sig")

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        if override or key not in os.environ:
            os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LSLM chat learning input runner")

    parser.add_argument(
        "--text",
        type=str,
        default="",
        help="Space-separated input text for fixed-input learning.",
    )
    parser.add_argument(
        "--words",
        nargs="*",
        default=None,
        help="Tokenized input for fixed-input learning.",
    )
    parser.add_argument(
        "--auto-input",
        action="store_true",
        help="Generate input sentences automatically using Gemini.",
    )
    parser.add_argument(
        "--input-generator-model",
        type=str,
        default="gemini-2.5-flash-lite",
        help="Gemini model for automatic input generation.",
    )
    parser.add_argument(
        "--lexicon",
        type=str,
        default="libs/dict.lsdx",
        help="Path to lexicon file.",
    )
    parser.add_argument(
        "--divergence-model",
        type=str,
        default="runtime/models/divergence_model.json",
        help="Path to divergence model state.",
    )
    parser.add_argument(
        "--convergence-model",
        type=str,
        default="runtime/models/convergence_model.json",
        help="Path to convergence model state.",
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-2.5-flash-lite",
        help="Gemini model for candidate evaluation.",
    )
    parser.add_argument(
        "--disable-evaluator",
        action="store_true",
        help="Disable external evaluator and use local fallback only.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Thought depth.",
    )
    parser.add_argument(
        "--branch",
        type=int,
        default=6,
        help="Default divergence branch size.",
    )
    parser.add_argument(
        "--final-branch",
        type=int,
        default=12,
        help="Final divergence branch size.",
    )
    parser.add_argument(
        "--max-recursive-steps",
        type=int,
        default=6,
        help="Maximum recursive thinking steps.",
    )
    parser.add_argument(
        "--accept-score-threshold",
        type=float,
        default=8.5,
        help="Accept threshold for recursive evaluation.",
    )
    parser.add_argument(
        "--infinity",
        action="store_true",
        help="Run infinitely until Ctrl+C.",
    )
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=0.0,
        help="Optional sleep time between episodes in infinity mode.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="runtime/logs/chat_learning.jsonl",
        help="JSONL log path.",
    )
    parser.add_argument(
        "--verbal-model-path",
        type=str,
        default="runtime/models/verbal_model.json",
        help="Verbal model JSON path.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    parser.add_argument(
        "--dotenv-path",
        type=str,
        default=".env",
        help="Path to .env file.",
    )
    parser.add_argument(
        "--dotenv-override",
        action="store_true",
        help="Override existing environment variables with values from .env.",
    )

    return parser.parse_args()


def resolve_fixed_input_tokens(args: argparse.Namespace) -> List[str]:
    if args.words:
        return [str(x).strip() for x in args.words if str(x).strip()]
    if args.text.strip():
        return [x for x in args.text.strip().split() if x.strip()]
    return []


def main() -> None:
    args = parse_args()

    load_dotenv_file(
        dotenv_path=args.dotenv_path,
        override=args.dotenv_override,
    )

    has_gemini = bool(os.getenv("GEMINI_API_KEY", "").strip())
    print(f"[ENV] GEMINI_API_KEY={'set' if has_gemini else 'missing'}", flush=True)

    if not args.auto_input:
        fixed_tokens = resolve_fixed_input_tokens(args)
        if not fixed_tokens:
            raise SystemExit("[ERROR] chat_learning requires --text or --words unless --auto-input is used")

    central = build_chat_learning_central(args)

    episode = 0
    try:
        while True:
            episode += 1
            print(f"[CHAT_INPUT] LOOP episode={episode}", flush=True)

            if args.auto_input:
                episode_tokens = central.generate_input_tokens()
                print(f"[CHAT_INPUT] AUTO_INPUT tokens={episode_tokens}", flush=True)
            else:
                episode_tokens = resolve_fixed_input_tokens(args)
                print(f"[CHAT_INPUT] FIXED_INPUT tokens={episode_tokens}", flush=True)

            result = central.run_once(
                input_tokens=episode_tokens,
                depth=args.depth,
            )

            print("\n[BEST TEXT]")
            print(result["best_text"])

            if not args.infinity:
                break

            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)

    except KeyboardInterrupt:
        print("[CHAT_INPUT] stopped by user", flush=True)


if __name__ == "__main__":
    main()