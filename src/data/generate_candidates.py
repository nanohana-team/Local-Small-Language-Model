import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests


DEFAULT_SYSTEM_PROMPT = (
    "あなたは日本語で自然に会話するAIです。"
    "簡潔で自然な返答をしてください。"
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL parse error at {path}:{line_no}: {e}") from e
    return items


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def safe_get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


@dataclass
class TeacherConfig:
    name: str
    api_base: str
    api_key_env: str
    model: str
    temperature: float
    max_tokens: int
    top_p: float = 1.0
    timeout_sec: int = 120

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TeacherConfig":
        required = ["name", "api_base", "api_key_env", "model"]
        for key in required:
            if key not in d:
                raise ValueError(f"Teacher config missing required key: {key}")

        return TeacherConfig(
            name=str(d["name"]),
            api_base=str(d["api_base"]).rstrip("/"),
            api_key_env=str(d["api_key_env"]),
            model=str(d["model"]),
            temperature=float(d.get("temperature", 0.8)),
            max_tokens=int(d.get("max_tokens", 256)),
            top_p=float(d.get("top_p", 1.0)),
            timeout_sec=int(d.get("timeout_sec", 120)),
        )


class OpenAICompatibleClient:
    def __init__(self, api_base: str, api_key: str, timeout_sec: int = 120):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.timeout_sec = timeout_sec

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float,
    ) -> Dict[str, Any]:
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.timeout_sec,
        )

        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            raise RuntimeError(
                f"API request failed: status={response.status_code} body={response.text}"
            ) from e

        return response.json()


def build_messages(
    prompt_item: Dict[str, Any],
    default_system_prompt: str,
) -> List[Dict[str, str]]:
    if "messages" in prompt_item:
        messages = prompt_item["messages"]
        if not isinstance(messages, list):
            raise ValueError("'messages' must be a list.")
        normalized: List[Dict[str, str]] = []
        for m in messages:
            role = str(m.get("role", "")).strip()
            content = str(m.get("content", "")).strip()
            if not role or not content:
                continue
            normalized.append({"role": role, "content": content})
        if not normalized:
            raise ValueError("Prompt item has empty messages.")
        return normalized

    user_text = str(prompt_item.get("input", "")).strip()
    if not user_text:
        raise ValueError("Prompt item must contain either 'messages' or non-empty 'input'.")

    system_prompt = str(prompt_item.get("system", "")).strip() or default_system_prompt
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]


def extract_user_input_text(prompt_item: Dict[str, Any]) -> str:
    if "input" in prompt_item and str(prompt_item["input"]).strip():
        return str(prompt_item["input"]).strip()

    if "messages" in prompt_item and isinstance(prompt_item["messages"], list):
        user_parts: List[str] = []
        for m in prompt_item["messages"]:
            if str(m.get("role", "")).strip() == "user":
                content = str(m.get("content", "")).strip()
                if content:
                    user_parts.append(content)
        return "\n".join(user_parts).strip()

    return ""


def load_teacher_configs(path: Path) -> List[TeacherConfig]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("Teacher config file must be a JSON array.")

    teachers = [TeacherConfig.from_dict(x) for x in raw]
    if not teachers:
        raise ValueError("Teacher config file is empty.")
    return teachers


def count_text_chars(text: str) -> int:
    return len(text or "")


def generate_one_candidate(
    teacher: TeacherConfig,
    messages: List[Dict[str, str]],
    prompt_item: Dict[str, Any],
    prompt_id: str,
    run_id: str,
    candidate_index: int,
    dry_run: bool = False,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    created_at = utc_now_iso()

    if dry_run:
        answer_text = f"[DRY RUN] {teacher.name} / {teacher.model} candidate={candidate_index}"
        elapsed_sec = time.perf_counter() - started_at
        return {
            "id": str(uuid.uuid4()),
            "run_id": run_id,
            "prompt_id": prompt_id,
            "candidate_index": candidate_index,
            "created_at": created_at,
            "teacher": {
                "name": teacher.name,
                "model": teacher.model,
                "api_base": teacher.api_base,
                "temperature": teacher.temperature,
                "max_tokens": teacher.max_tokens,
                "top_p": teacher.top_p,
            },
            "input": {
                "text": extract_user_input_text(prompt_item),
                "messages": messages,
                "meta": prompt_item.get("meta", {}),
            },
            "output": {
                "text": answer_text,
                "finish_reason": "dry_run",
                "usage": {},
                "char_count": count_text_chars(answer_text),
            },
            "timing": {
                "elapsed_sec": round(elapsed_sec, 4),
            },
            "error": None,
        }

    api_key = safe_get_env(teacher.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Environment variable '{teacher.api_key_env}' is not set for teacher '{teacher.name}'."
        )

    client = OpenAICompatibleClient(
        api_base=teacher.api_base,
        api_key=api_key,
        timeout_sec=teacher.timeout_sec,
    )

    result = client.chat_completion(
        model=teacher.model,
        messages=messages,
        temperature=teacher.temperature,
        max_tokens=teacher.max_tokens,
        top_p=teacher.top_p,
    )

    choices = result.get("choices", [])
    if not choices:
        raise RuntimeError("API response does not contain choices.")

    first = choices[0]
    message = first.get("message", {}) or {}
    answer_text = str(message.get("content", "")).strip()
    finish_reason = first.get("finish_reason")

    elapsed_sec = time.perf_counter() - started_at

    return {
        "id": str(uuid.uuid4()),
        "run_id": run_id,
        "prompt_id": prompt_id,
        "candidate_index": candidate_index,
        "created_at": created_at,
        "teacher": {
            "name": teacher.name,
            "model": teacher.model,
            "api_base": teacher.api_base,
            "temperature": teacher.temperature,
            "max_tokens": teacher.max_tokens,
            "top_p": teacher.top_p,
        },
        "input": {
            "text": extract_user_input_text(prompt_item),
            "messages": messages,
            "meta": prompt_item.get("meta", {}),
        },
        "output": {
            "text": answer_text,
            "finish_reason": finish_reason,
            "usage": result.get("usage", {}),
            "char_count": count_text_chars(answer_text),
        },
        "timing": {
            "elapsed_sec": round(elapsed_sec, 4),
        },
        "error": None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate candidate responses from multiple teacher models."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/prompts/input_prompts.jsonl",
        help="Path to input prompts JSONL.",
    )
    parser.add_argument(
        "--teacher-config",
        type=str,
        default="config/teachers.json",
        help="Path to teacher config JSON.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/candidates/candidates.jsonl",
        help="Path to output JSONL file.",
    )
    parser.add_argument(
        "--default-system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="Default system prompt if input item only has 'input'.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit of prompt items to process. 0 means all.",
    )
    parser.add_argument(
        "--repeats-per-teacher",
        type=int,
        default=1,
        help="How many candidates to generate per teacher for each prompt.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to output file. If not set, overwrite output file.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing even if one teacher call fails.",
    )
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=0.0,
        help="Sleep seconds between requests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call APIs. Emit fake outputs for pipeline testing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_path = Path(args.input_file)
    teacher_config_path = Path(args.teacher_config)
    output_path = Path(args.output_file)

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        return 1

    if not teacher_config_path.exists():
        print(f"[ERROR] Teacher config not found: {teacher_config_path}", file=sys.stderr)
        return 1

    prompts = load_jsonl(input_path)
    teachers = load_teacher_configs(teacher_config_path)

    if args.limit > 0:
        prompts = prompts[: args.limit]

    if not args.append:
        ensure_parent_dir(output_path)
        output_path.write_text("", encoding="utf-8")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]

    total_prompt_count = len(prompts)
    total_teacher_count = len(teachers)
    total_expected = total_prompt_count * total_teacher_count * args.repeats_per_teacher

    print("[INFO] generate_candidates started")
    print(f"[INFO] run_id={run_id}")
    print(f"[INFO] prompts={total_prompt_count}")
    print(f"[INFO] teachers={total_teacher_count}")
    print(f"[INFO] repeats_per_teacher={args.repeats_per_teacher}")
    print(f"[INFO] expected_candidates={total_expected}")
    print(f"[INFO] output_file={output_path}")

    written_count = 0
    error_count = 0

    for prompt_idx, prompt_item in enumerate(prompts, start=1):
        prompt_id = str(prompt_item.get("id") or f"prompt-{prompt_idx:06d}")
        try:
            messages = build_messages(prompt_item, args.default_system_prompt)
        except Exception as e:
            error_count += 1
            print(
                f"[ERROR] Failed to build messages for prompt_id={prompt_id}: {e}",
                file=sys.stderr,
            )
            if not args.continue_on_error:
                return 1
            continue

        print(f"[INFO] prompt {prompt_idx}/{total_prompt_count} prompt_id={prompt_id}")

        for teacher in teachers:
            for repeat_idx in range(args.repeats_per_teacher):
                candidate_index = repeat_idx + 1
                try:
                    row = generate_one_candidate(
                        teacher=teacher,
                        messages=messages,
                        prompt_item=prompt_item,
                        prompt_id=prompt_id,
                        run_id=run_id,
                        candidate_index=candidate_index,
                        dry_run=args.dry_run,
                    )
                    append_jsonl(output_path, [row])
                    written_count += 1

                    print(
                        "[OK] "
                        f"prompt_id={prompt_id} "
                        f"teacher={teacher.name} "
                        f"model={teacher.model} "
                        f"candidate={candidate_index} "
                        f"chars={row['output']['char_count']} "
                        f"time={row['timing']['elapsed_sec']}s"
                    )
                except Exception as e:
                    error_count += 1
                    error_row = {
                        "id": str(uuid.uuid4()),
                        "run_id": run_id,
                        "prompt_id": prompt_id,
                        "candidate_index": candidate_index,
                        "created_at": utc_now_iso(),
                        "teacher": {
                            "name": teacher.name,
                            "model": teacher.model,
                            "api_base": teacher.api_base,
                            "temperature": teacher.temperature,
                            "max_tokens": teacher.max_tokens,
                            "top_p": teacher.top_p,
                        },
                        "input": {
                            "text": extract_user_input_text(prompt_item),
                            "messages": messages,
                            "meta": prompt_item.get("meta", {}),
                        },
                        "output": None,
                        "timing": {},
                        "error": {
                            "message": str(e),
                            "type": type(e).__name__,
                        },
                    }
                    append_jsonl(output_path, [error_row])

                    print(
                        "[ERROR] "
                        f"prompt_id={prompt_id} "
                        f"teacher={teacher.name} "
                        f"model={teacher.model} "
                        f"candidate={candidate_index} "
                        f"error={e}",
                        file=sys.stderr,
                    )
                    if not args.continue_on_error:
                        return 1

                if args.sleep_sec > 0:
                    time.sleep(args.sleep_sec)

    print("[INFO] generate_candidates finished")
    print(f"[INFO] written_count={written_count}")
    print(f"[INFO] error_count={error_count}")
    print(f"[INFO] output_file={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())