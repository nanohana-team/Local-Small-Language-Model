from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from dotenv import load_dotenv
from google import genai
from google.genai import types


DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clamp_score(x: Any) -> float:
    try:
        value = float(x)
    except Exception:
        return 0.0
    value = max(0.0, min(1.0, value))
    return round(value, 4)


def build_eval_prompt(session: Dict[str, Any], learning_name: str) -> str:
    lines: List[str] = []
    lines.append("あなたは会話セッションを評価する厳格なレビュアーです。")
    lines.append("出力は必ずJSONのみとし、余計な説明文は出力しないこと。")
    lines.append("目的は learning モデルを改善するために、会話全体と各発話を評価することです。")
    lines.append("")
    lines.append("必須要件:")
    lines.append("- session 全体の model_ranking を返す")
    lines.append("- turn_scores は learning モデルの発話を必ず全件含める")
    lines.append("- verdict は good_for_sft / usable / bad / unsafe のいずれか")
    lines.append("- preferences には chosen と rejected の比較ペアを必要なだけ入れる")
    lines.append("- 0.0000〜1.0000 の範囲で採点する")
    lines.append("")
    lines.append(f"learning model speaker name: {learning_name}")
    lines.append("")
    lines.append("session:")
    lines.append(json.dumps(session, ensure_ascii=False, indent=2))
    return "\n".join(lines)


def normalize_result(raw: Dict[str, Any], session: Dict[str, Any], learning_name: str) -> Dict[str, Any]:
    turns = session.get("turns", [])
    learning_turn_map = {
        int(t["turn_index"]): t
        for t in turns
        if str(t.get("speaker")) == learning_name
    }

    turn_scores = raw.get("turn_scores", [])
    if not isinstance(turn_scores, list):
        turn_scores = []

    normalized_turn_scores: List[Dict[str, Any]] = []
    existing_indices = set()

    for item in turn_scores:
        if not isinstance(item, dict):
            continue
        turn_index = item.get("turn_index")
        if not isinstance(turn_index, int) or turn_index not in learning_turn_map:
            continue
        existing_indices.add(turn_index)
        normalized_turn_scores.append(
            {
                "turn_index": turn_index,
                "speaker": learning_name,
                "intent_fit": clamp_score(item.get("intent_fit")),
                "naturalness": clamp_score(item.get("naturalness")),
                "consistency": clamp_score(item.get("consistency")),
                "informativeness": clamp_score(item.get("informativeness")),
                "conciseness": clamp_score(item.get("conciseness")),
                "safety": clamp_score(item.get("safety")),
                "completeness": clamp_score(item.get("completeness")),
                "language_purity": clamp_score(item.get("language_purity")),
                "overall": clamp_score(item.get("overall")),
                "verdict": str(item.get("verdict") or "usable"),
                "reason_short": str(item.get("reason_short") or ""),
                "text": learning_turn_map[turn_index]["text"],
            }
        )

    for turn_index, turn in sorted(learning_turn_map.items()):
        if turn_index in existing_indices:
            continue
        normalized_turn_scores.append(
            {
                "turn_index": turn_index,
                "speaker": learning_name,
                "intent_fit": 0.0,
                "naturalness": 0.0,
                "consistency": 0.0,
                "informativeness": 0.0,
                "conciseness": 0.0,
                "safety": 0.0,
                "completeness": 0.0,
                "language_purity": 0.0,
                "overall": 0.0,
                "verdict": "bad",
                "reason_short": "missing_from_gemini_output",
                "text": turn["text"],
            }
        )

    ranking = raw.get("model_ranking", [])
    if not isinstance(ranking, list):
        ranking = []
    speakers = []
    for t in turns:
        speaker = str(t.get("speaker"))
        if speaker not in speakers:
            speakers.append(speaker)
    ranking = [x for x in ranking if isinstance(x, str) and x in speakers]
    for speaker in speakers:
        if speaker not in ranking:
            ranking.append(speaker)

    preferences = raw.get("preferences", [])
    if not isinstance(preferences, list):
        preferences = []

    clean_preferences = []
    for item in preferences:
        if not isinstance(item, dict):
            continue
        clean_preferences.append(
            {
                "turn_index": int(item.get("turn_index", -1)),
                "chosen_speaker": str(item.get("chosen_speaker") or ""),
                "chosen_text": str(item.get("chosen_text") or ""),
                "rejected_speaker": str(item.get("rejected_speaker") or ""),
                "rejected_text": str(item.get("rejected_text") or ""),
                "reason_short": str(item.get("reason_short") or ""),
            }
        )

    return {
        "session_id": str(raw.get("session_id") or session.get("session_id")),
        "prompt_id": str(session.get("prompt_id") or ""),
        "input": str(session.get("input") or ""),
        "model_ranking": ranking,
        "turn_scores": sorted(normalized_turn_scores, key=lambda x: x["turn_index"]),
        "preferences": clean_preferences,
        "session_meta": session.get("meta", {}),
        "raw_session": session,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate conversation sessions with Gemini and emit learning-focused scores."
    )
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--learning-name", type=str, default="learning_gemma")
    parser.add_argument("--schema-file", type=str, default="config/gemini_eval_schema.json")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--env-file", type=str, default=".env")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(args.env_file)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    sessions = load_jsonl(Path(args.input_file))
    schema = load_json(Path(args.schema_file))
    client = genai.Client(api_key=api_key)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")

    print("[INFO] evaluate_conversations_with_gemini started")
    print(f"[INFO] sessions={len(sessions)}")

    for idx, session in enumerate(sessions, start=1):
        prompt = build_eval_prompt(session, args.learning_name)
        response = client.models.generate_content(
            model=args.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        text = response.text
        if not text:
            raise RuntimeError("Gemini returned empty response text.")
        raw = json.loads(text)
        result = normalize_result(raw, session, args.learning_name)
        append_jsonl(output_path, [result])
        print(
            f"[OK] session={idx}/{len(sessions)} "
            f"session_id={result['session_id']} "
            f"turn_scores={len(result['turn_scores'])} "
            f"preferences={len(result['preferences'])}"
        )

    print("[INFO] evaluate_conversations_with_gemini finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
