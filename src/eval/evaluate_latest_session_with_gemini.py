from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = BASE_DIR / "src" / "chat" / "logs" / "sessions"
DEFAULT_EVAL_DIR = BASE_DIR / "src" / "chat" / "logs" / "evals"
DEFAULT_ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=DEFAULT_ENV_PATH)

POINT_LABELS = [
    "intent_fit",
    "naturalness",
    "consistency",
    "informativeness",
    "conciseness",
    "safety",
    "completeness",
    "language_purity",
]

USER_MARKERS = ("Human:", "User:", "ユーザー:", "human:", "user:", "assistant:", "Assistant:")
MAX_HISTORY_ITEMS_FOR_PROMPT = 80


@dataclass
class MessageRecord:
    index: int
    model: str
    port: Optional[int]
    text: str
    cleaned_text: str


@dataclass
class SessionPayload:
    session_path: Path
    session_id: str
    updated_at: str
    context_text: str
    messages: List[MessageRecord]
    raw_preview: str


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def find_latest_session_file(log_dir: Path) -> Path:
    if not log_dir.exists():
        raise FileNotFoundError(f"log directory not found: {log_dir}")

    files = [p for p in log_dir.rglob("*.json") if p.is_file()]
    if not files:
        raise FileNotFoundError(f"no json files found under: {log_dir}")

    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def try_parse_json(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except Exception:
        return None


def try_parse_jsonl(text: str) -> Optional[List[Any]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None

    parsed: List[Any] = []
    for line in lines:
        try:
            parsed.append(json.loads(line))
        except Exception:
            return None
    return parsed


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, indent=2)


def clamp_point(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return round(x, 4)


def round4(value: float) -> float:
    return round(float(value), 4)


def extract_last_user_message_from_text(text: str) -> Optional[str]:
    if not text:
        return None

    patterns = [
        r"(?:^|\n)\s*Human:\s*(.+)$",
        r"(?:^|\n)\s*User:\s*(.+)$",
        r"(?:^|\n)\s*ユーザー:\s*(.+)$",
        r"(?:^|\n)\s*human:\s*(.+)$",
        r"(?:^|\n)\s*user:\s*(.+)$",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.DOTALL)
        if m:
            candidate = m.group(1).strip()
            if candidate:
                return candidate
    return None


def strip_embedded_noise(text: str, known_model_names: List[str]) -> str:
    if not text:
        return text

    markers = list(USER_MARKERS)
    for name in known_model_names:
        if name:
            markers.append(f"{name}:")
            markers.append(f"{name}：")

    cut_index = len(text)
    for marker in markers:
        idx = text.find(marker)
        if idx != -1:
            cut_index = min(cut_index, idx)

    cleaned = text[:cut_index].strip()
    return cleaned if cleaned else text.strip()


def extract_recent_context(messages: List[MessageRecord], limit: int = 6) -> str:
    if not messages:
        return ""
    selected = messages[-limit:]
    lines = [f"{m.model}: {m.cleaned_text}" for m in selected if m.cleaned_text]
    return "\n".join(lines).strip()


def load_history_session(obj: Dict[str, Any], session_path: Path) -> Optional[SessionPayload]:
    history = obj.get("history")
    if not isinstance(history, list):
        return None

    known_model_names: List[str] = []
    for item in history:
        if isinstance(item, dict):
            name = str(item.get("model") or item.get("model_name") or "").strip()
            if name and name not in known_model_names:
                known_model_names.append(name)

    messages: List[MessageRecord] = []
    discovered_user_inputs: List[str] = []

    for idx, item in enumerate(history):
        if not isinstance(item, dict):
            continue

        model = str(item.get("model") or item.get("model_name") or "unknown_model")
        port = item.get("port")
        raw_text = normalize_text(item.get("text") or item.get("content") or "")
        if not raw_text:
            continue

        user_msg = extract_last_user_message_from_text(raw_text)
        if user_msg:
            discovered_user_inputs.append(user_msg)

        cleaned_text = strip_embedded_noise(raw_text, known_model_names)
        if not cleaned_text:
            continue

        messages.append(
            MessageRecord(
                index=idx,
                model=model,
                port=port if isinstance(port, int) else None,
                text=raw_text,
                cleaned_text=cleaned_text,
            )
        )

    if not messages:
        return None

    recent_context = extract_recent_context(messages)
    last_user_input = discovered_user_inputs[-1] if discovered_user_inputs else ""

    if last_user_input and recent_context:
        context_text = f"Recent context:\n{recent_context}\n\nLast discovered user input:\n{last_user_input}"
    elif last_user_input:
        context_text = f"Last discovered user input:\n{last_user_input}"
    else:
        context_text = f"Recent context:\n{recent_context}" if recent_context else ""

    preview_obj = {
        "session_id": obj.get("session_id"),
        "updated_at": obj.get("updated_at"),
        "message_count": len(messages),
        "models": sorted(list({m.model for m in messages})),
        "context_text": context_text,
        "last_messages": [
            {
                "index": m.index,
                "model": m.model,
                "port": m.port,
                "cleaned_text": m.cleaned_text,
            }
            for m in messages[-10:]
        ],
    }

    return SessionPayload(
        session_path=session_path,
        session_id=str(obj.get("session_id") or session_path.stem),
        updated_at=str(obj.get("updated_at") or ""),
        context_text=context_text,
        messages=messages,
        raw_preview=json.dumps(preview_obj, ensure_ascii=False, indent=2)[:4000],
    )


def load_session_payload(session_path: Path) -> SessionPayload:
    text = read_text_file(session_path)

    obj = try_parse_json(text)
    if isinstance(obj, dict):
        payload = load_history_session(obj, session_path)
        if payload is not None:
            return payload

    objl = try_parse_jsonl(text)
    if objl is not None:
        pseudo_obj = {"session_id": session_path.stem, "updated_at": "", "history": objl}
        payload = load_history_session(pseudo_obj, session_path)
        if payload is not None:
            return payload

    raise ValueError(f"unsupported session log format: {session_path}")


def build_eval_prompt(payload: SessionPayload) -> str:
    lines: List[str] = []
    lines.append("あなたは会話ログ全体を評価する厳格なレビュアーです。")
    lines.append("出力は必ずJSONのみとし、JSON以外の文字は一切出力しないこと。")
    lines.append("説明文、要約、コメント、理由文は禁止。")
    lines.append("")
    lines.append("目的:")
    lines.append("- 会話履歴全体を読み、モデルごとの総合得点をつけること")
    lines.append("- 候補単発ではなく、全発話を通した傾向を評価すること")
    lines.append("- モデル数は固定ではない。与えられた全モデルを評価すること")
    lines.append("")
    lines.append("評価軸は以下の8個で固定すること。")
    lines.append("1. intent_fit: 会話文脈やユーザー意図との適合度")
    lines.append("2. naturalness: 日本語としての自然さ")
    lines.append("3. consistency: 内容の一貫性と破綻の少なさ")
    lines.append("4. informativeness: 有用な情報の量")
    lines.append("5. conciseness: 無駄の少なさ")
    lines.append("6. safety: 不適切・危険な内容の少なさ")
    lines.append("7. completeness: 文が途中で切れず完結しているか")
    lines.append("8. language_purity: 不要な英単語混入や不自然な言語混在の少なさ")
    lines.append("")
    lines.append("採点ルール:")
    lines.append("- 各モデルについて、全発話を総合して 8 軸を 0.0000〜1.0000 で採点する")
    lines.append("- 1.0000 が最良、0.0000 が最悪")
    lines.append("- points の桁数は必ず小数点以下4桁")
    lines.append("- overall は points の総合評価として 0.0000〜1.0000")
    lines.append("- count はそのモデルの評価対象発話数")
    lines.append("- ranking は overall の高い順")
    lines.append("- モデル数が何件でも全件返すこと")
    lines.append("")
    lines.append("出力形式は必ずこれに厳密一致:")
    lines.append(
        '{"ranking":["model_a","model_b"],"models":[{"model":"model_a","points":[0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000],"overall":0.0000,"count":0},{"model":"model_b","points":[0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000],"overall":0.0000,"count":0}]}'
    )
    lines.append("")
    lines.append("補足:")
    lines.append("- 途中で切れた文、他モデル名や Human: の混入、不自然な話題逸脱は減点対象")
    lines.append("- 発話数が少なくても評価は行う")
    lines.append("")
    lines.append("直前文脈または補助情報:")
    lines.append(payload.context_text if payload.context_text else "(context not found)")
    lines.append("")
    lines.append("会話履歴:")
    for m in payload.messages[-MAX_HISTORY_ITEMS_FOR_PROMPT:]:
        lines.append(f"[{m.index}] model={m.model} port={m.port}")
        lines.append(m.cleaned_text)
        lines.append("")

    lines.append("raw_preview:")
    lines.append(payload.raw_preview)
    return "\n".join(lines)


def get_response_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "ranking": {
                "type": "array",
                "items": {"type": "string"},
            },
            "models": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "model": {"type": "string"},
                        "points": {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                        "overall": {"type": "number"},
                        "count": {"type": "integer"},
                    },
                    "required": ["model", "points", "overall", "count"],
                },
            },
        },
        "required": ["ranking", "models"],
    }


def build_model_counts(messages: List[MessageRecord]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for m in messages:
        counts[m.model] = counts.get(m.model, 0) + 1
    return counts


def postprocess_evaluation_data(data: Dict[str, Any], payload: SessionPayload) -> Dict[str, Any]:
    model_counts = build_model_counts(payload.messages)
    valid_models = list(model_counts.keys())

    ranking = data.get("ranking", [])
    if not isinstance(ranking, list):
        ranking = []
    ranking = [x for x in ranking if isinstance(x, str) and x in model_counts]
    for model_name in valid_models:
        if model_name not in ranking:
            ranking.append(model_name)

    model_entries = data.get("models", [])
    if not isinstance(model_entries, list):
        model_entries = []

    raw_map: Dict[str, Dict[str, Any]] = {}
    for item in model_entries:
        if not isinstance(item, dict):
            continue
        model_name = item.get("model")
        if not isinstance(model_name, str) or model_name not in model_counts:
            continue

        points = item.get("points", [])
        if not isinstance(points, list):
            points = []
        cleaned_points = [clamp_point(x) for x in points]
        if len(cleaned_points) < len(POINT_LABELS):
            cleaned_points.extend([0.0] * (len(POINT_LABELS) - len(cleaned_points)))
        elif len(cleaned_points) > len(POINT_LABELS):
            cleaned_points = cleaned_points[: len(POINT_LABELS)]

        overall = clamp_point(item.get("overall", 0.0))
        count = item.get("count", model_counts[model_name])
        if not isinstance(count, int):
            count = model_counts[model_name]

        raw_map[model_name] = {
            "model": model_name,
            "points": cleaned_points,
            "overall": overall,
            "count": count,
        }

    normalized_models: List[Dict[str, Any]] = []
    for model_name in valid_models:
        entry = raw_map.get(
            model_name,
            {
                "model": model_name,
                "points": [0.0] * len(POINT_LABELS),
                "overall": 0.0,
                "count": model_counts[model_name],
            },
        )
        normalized_models.append(entry)

    score_map = {entry["model"]: entry["overall"] for entry in normalized_models}
    ranking = sorted(
        ranking,
        key=lambda name: (score_map.get(name, 0.0), model_counts.get(name, 0)),
        reverse=True,
    )

    return {
        "point_labels": POINT_LABELS,
        "ranking": ranking,
        "models": normalized_models,
        "session_id": payload.session_id,
        "updated_at": payload.updated_at,
    }


def evaluate_with_gemini(
    payload: SessionPayload,
    model_name: str,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    if not api_key:
        raise RuntimeError(
            f"GEMINI_API_KEY is not set. Please check your .env file: {DEFAULT_ENV_PATH}"
        )

    client = genai.Client(api_key=api_key)
    prompt = build_eval_prompt(payload)

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=get_response_schema(),
        ),
    )

    text = response.text
    if not text:
        raise RuntimeError("Gemini returned empty response text.")

    data = json.loads(text)
    return postprocess_evaluation_data(data, payload)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read the latest session log and evaluate all models over the full conversation with Gemini."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="directory containing session logs",
    )
    parser.add_argument(
        "--session-file",
        type=Path,
        default=None,
        help="specific session file to evaluate; if omitted, latest file under --log-dir is used",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Gemini model name",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output JSON path; default is src/chat/logs/evals/<session_stem>.eval.json",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="print debug information to stderr",
    )
    args = parser.parse_args()

    try:
        log_dir = args.log_dir.resolve()
        session_path = args.session_file.resolve() if args.session_file else find_latest_session_file(log_dir)
        payload = load_session_payload(session_path)

        if args.debug:
            model_counts = build_model_counts(payload.messages)
            print(f"[DEBUG] session_path={session_path}", file=sys.stderr)
            print(f"[DEBUG] session_id={payload.session_id}", file=sys.stderr)
            print(f"[DEBUG] message_count={len(payload.messages)}", file=sys.stderr)
            print(f"[DEBUG] model_count={len(model_counts)}", file=sys.stderr)
            for model_name, count in sorted(model_counts.items(), key=lambda x: (-x[1], x[0])):
                print(f"[DEBUG] model={model_name} count={count}", file=sys.stderr)

        result = evaluate_with_gemini(
            payload=payload,
            model_name=args.model,
            api_key=os.getenv("GEMINI_API_KEY"),
        )

        pretty = json.dumps(result, ensure_ascii=False, indent=2)
        print(pretty)

        out_path = args.out.resolve() if args.out else (DEFAULT_EVAL_DIR / f"{session_path.stem}.eval.json")
        ensure_parent_dir(out_path)
        out_path.write_text(pretty, encoding="utf-8")
        print(f"\n[INFO] saved: {out_path}", file=sys.stderr)

        return 0

    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())