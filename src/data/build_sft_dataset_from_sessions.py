from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_SYSTEM_PROMPT = (
    "あなたは日本語で自然に会話する小型AIです。"
    "簡潔で自然な返答を心がけてください。"
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL parse error at {path}:{line_no}: {e}") from e
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def verdict_is_accepted(verdict: str) -> bool:
    return verdict in {"good_for_sft", "usable"}


def build_messages_from_prefix(session: Dict[str, Any], turn_index: int) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": str(session.get("input", ""))},
    ]
    for turn in session.get("raw_session", {}).get("turns", []):
        if int(turn["turn_index"]) >= turn_index:
            break
        messages.append({"role": "assistant", "content": str(turn.get("text", ""))})
    return messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SFT dataset from Gemini-scored multi-model sessions."
    )
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--train-output-file", required=True)
    parser.add_argument("--eval-output-file", required=True)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--min-overall", type=float, default=0.70)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    sessions = load_jsonl(Path(args.input_file))
    sft_rows: List[Dict[str, Any]] = []

    for session in sessions:
        for item in session.get("turn_scores", []):
            overall = float(item.get("overall", 0.0))
            verdict = str(item.get("verdict", ""))
            text = str(item.get("text", "")).strip()
            turn_index = int(item.get("turn_index", -1))
            if turn_index < 0 or not text:
                continue
            if overall < args.min_overall:
                continue
            if not verdict_is_accepted(verdict):
                continue

            messages = build_messages_from_prefix(session, turn_index)
            messages.append({"role": "assistant", "content": text})

            sft_rows.append(
                {
                    "messages": messages,
                    "meta": {
                        "session_id": session.get("session_id"),
                        "prompt_id": session.get("prompt_id"),
                        "turn_index": turn_index,
                        "overall": overall,
                        "verdict": verdict,
                    },
                }
            )

    random.shuffle(sft_rows)

    if not sft_rows:
        raise RuntimeError("No SFT rows were produced. Lower --min-overall or inspect evaluation output.")

    split = int(len(sft_rows) * (1 - args.eval_ratio))
    if split <= 0:
        split = max(1, len(sft_rows) - 1)
    if split >= len(sft_rows):
        split = len(sft_rows)

    train_rows = sft_rows[:split]
    eval_rows = sft_rows[split:] if split < len(sft_rows) else []

    write_jsonl(Path(args.train_output_file), train_rows)
    write_jsonl(Path(args.eval_output_file), eval_rows)

    print(f"[INFO] sft_train={len(train_rows)} sft_eval={len(eval_rows)} total={len(sft_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
