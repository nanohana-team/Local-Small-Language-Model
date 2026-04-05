from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.apps.chat_learning_central import (
    ChatLearningCentral,
    build_chat_learning_central,
)
from src.utils.logging import (
    load_dotenv_file,
)


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
        "--target-text",
        type=str,
        default="",
        help="Teacher target surface text.",
    )
    parser.add_argument(
        "--target-words",
        nargs="*",
        default=None,
        help="Teacher target tokens.",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="",
        help="Optional JSON/JSONL file for fixed supervised episodes.",
    )
    parser.add_argument(
        "--auto-teacher",
        action="store_true",
        help="Generate input/target pairs automatically using Gemini.",
    )
    parser.add_argument(
        "--teacher-generator-model",
        type=str,
        default="gemini-2.5-flash-lite",
        help="Gemini model for automatic teacher pair generation.",
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
        default=0.85,
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


def _clean_tokens(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    return [str(x).strip() for x in values if str(x).strip()]


def resolve_fixed_input_tokens(args: argparse.Namespace) -> List[str]:
    if args.words:
        return [str(x).strip() for x in args.words if str(x).strip()]
    if args.text.strip():
        return [x for x in args.text.strip().split() if x.strip()]
    return []


def resolve_fixed_target_tokens(args: argparse.Namespace) -> List[str]:
    if args.target_words:
        return [str(x).strip() for x in args.target_words if str(x).strip()]
    return []


def _normalize_episode_row(row: Any) -> Optional[Dict[str, Any]]:
    if isinstance(row, list):
        input_tokens = _clean_tokens(row)
        if not input_tokens:
            return None
        return {
            "input_tokens": input_tokens,
            "target_tokens": [],
            "target_text": "",
        }

    if not isinstance(row, dict):
        return None

    input_tokens: List[str] = []
    target_tokens: List[str] = []
    target_text = ""

    if "input_tokens" in row:
        input_tokens = _clean_tokens(row.get("input_tokens"))
    elif "tokens" in row:
        input_tokens = _clean_tokens(row.get("tokens"))
    elif isinstance(row.get("input_text"), str):
        input_tokens = [x for x in str(row["input_text"]).strip().split() if x.strip()]
    elif isinstance(row.get("text"), str):
        input_tokens = [x for x in str(row["text"]).strip().split() if x.strip()]

    if "target_tokens" in row:
        target_tokens = _clean_tokens(row.get("target_tokens"))

    if isinstance(row.get("target_text"), str):
        target_text = str(row.get("target_text", "")).strip()

    if not input_tokens:
        return None

    return {
        "input_tokens": input_tokens,
        "target_tokens": target_tokens,
        "target_text": target_text,
    }


def read_episode_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"input file not found: {path}")

    rows: List[Dict[str, Any]] = []
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception as exc:
                    print(f"[WARN] failed to parse jsonl line={line_no}: {exc}", flush=True)
                    continue

                normalized = _normalize_episode_row(row)
                if normalized is not None:
                    rows.append(normalized)

    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        if isinstance(obj, list):
            for idx, row in enumerate(obj, start=1):
                normalized = _normalize_episode_row(row)
                if normalized is not None:
                    rows.append(normalized)
                else:
                    print(f"[WARN] skipped invalid json row index={idx}", flush=True)

    else:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                tokens = [x for x in line.strip().split() if x.strip()]
                if not tokens:
                    continue
                rows.append(
                    {
                        "input_tokens": tokens,
                        "target_tokens": [],
                        "target_text": "",
                    }
                )

    return rows


def main() -> None:
    args = parse_args()

    load_dotenv_file(
        dotenv_path=args.dotenv_path,
        override=args.dotenv_override,
    )

    has_gemini = bool(os.getenv("GEMINI_API_KEY", "").strip())
    print(f"[ENV] GEMINI_API_KEY={'set' if has_gemini else 'missing'}", flush=True)

    central = build_chat_learning_central(args)

    fixed_rows: List[Dict[str, Any]] = []
    if args.input_file.strip():
        fixed_rows = read_episode_rows(Path(args.input_file))
        if not fixed_rows:
            raise SystemExit("[ERROR] input_file was provided but no valid episodes were found")
        print(f"[CHAT_INPUT] loaded dataset rows={len(fixed_rows)} from {args.input_file}", flush=True)

    elif not args.auto_teacher:
        fixed_tokens = resolve_fixed_input_tokens(args)
        fixed_target_tokens = resolve_fixed_target_tokens(args)
        fixed_target_text = str(args.target_text or "").strip()

        if not fixed_tokens:
            raise SystemExit(
                "[ERROR] chat_learning requires --text or --words unless --auto-teacher is used"
            )

        fixed_rows = [
            {
                "input_tokens": fixed_tokens,
                "target_tokens": fixed_target_tokens,
                "target_text": fixed_target_text,
            }
        ]

    episode = 0
    dataset_index = 0

    try:
        while True:
            episode += 1
            print(f"[CHAT_INPUT] LOOP episode={episode}", flush=True)

            if args.auto_teacher:
                pair = central.generate_teacher_pair()
                episode_tokens = [str(x).strip() for x in pair.get("input_tokens", []) if str(x).strip()]
                target_tokens = [str(x).strip() for x in pair.get("target_tokens", []) if str(x).strip()]
                target_text = str(pair.get("target_text", "")).strip()

                print(
                    f"[CHAT_INPUT] AUTO_TEACHER input={episode_tokens} "
                    f"target_tokens={target_tokens} "
                    f"target_text={json.dumps(target_text, ensure_ascii=False)}",
                    flush=True,
                )
            else:
                row = fixed_rows[dataset_index % len(fixed_rows)]
                dataset_index += 1

                episode_tokens = [str(x).strip() for x in row.get("input_tokens", []) if str(x).strip()]
                target_tokens = [str(x).strip() for x in row.get("target_tokens", []) if str(x).strip()]
                target_text = str(row.get("target_text", "")).strip()

                print(
                    f"[CHAT_INPUT] FIXED_INPUT tokens={episode_tokens} "
                    f"target_tokens={target_tokens} "
                    f"target_text={json.dumps(target_text, ensure_ascii=False)}",
                    flush=True,
                )

            result = central.run_once(
                input_tokens=episode_tokens,
                depth=args.depth,
                target_tokens=target_tokens,
                target_text=target_text,
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