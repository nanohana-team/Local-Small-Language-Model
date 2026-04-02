from __future__ import annotations

import argparse
import json
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


def build_prompt_messages(session: Dict[str, Any], turn_index: int) -> List[Dict[str, str]]:
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
        description="Build preference pairs from Gemini-scored multi-model sessions."
    )
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--learning-name", default="learning_gemma")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sessions = load_jsonl(Path(args.input_file))
    rows: List[Dict[str, Any]] = []

    for session in sessions:
        for pref in session.get("preferences", []):
            chosen_text = str(pref.get("chosen_text", "")).strip()
            rejected_text = str(pref.get("rejected_text", "")).strip()
            turn_index = int(pref.get("turn_index", -1))
            if turn_index < 0 or not chosen_text or not rejected_text:
                continue
            messages = build_prompt_messages(session, turn_index)
            rows.append(
                {
                    "messages": messages,
                    "chosen": chosen_text,
                    "rejected": rejected_text,
                    "meta": {
                        "session_id": session.get("session_id"),
                        "prompt_id": session.get("prompt_id"),
                        "turn_index": turn_index,
                        "chosen_speaker": pref.get("chosen_speaker"),
                        "rejected_speaker": pref.get("rejected_speaker"),
                        "reason_short": pref.get("reason_short", ""),
                    },
                }
            )

    write_jsonl(Path(args.output_file), rows)
    print(f"[INFO] dpo_rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
