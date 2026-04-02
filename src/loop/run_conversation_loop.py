from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.inference.local_student_client import LearningConfig, LocalStudentClient
from src.inference.teacher_client import TeacherConfig
from src.loop.run_selfplay_session import ConversationLoopConfig, run_single_session


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-turn conversation sessions with teacher models and one learning model."
    )
    parser.add_argument("--prompts-file", type=str, required=True)
    parser.add_argument("--teachers-config", type=str, required=True)
    parser.add_argument("--learning-config", type=str, required=True)
    parser.add_argument("--loop-config", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    prompts = load_jsonl(Path(args.prompts_file))
    teachers_raw = load_json(Path(args.teachers_config))
    learning_raw = load_json(Path(args.learning_config))
    loop_raw = load_json(Path(args.loop_config))

    if args.limit > 0:
        prompts = prompts[: args.limit]

    teachers = {str(x["name"]): TeacherConfig.from_dict(x) for x in teachers_raw}
    learning_cfg = LearningConfig.from_dict(learning_raw)
    loop_cfg = ConversationLoopConfig.from_dict(loop_raw)

    missing = [name for name in loop_cfg.participants if name != learning_cfg.name and name not in teachers]
    if missing:
        raise ValueError(f"Participants not found in teachers config: {missing}")

    learning_client = LocalStudentClient(learning_cfg)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")

    print("[INFO] run_conversation_loop started")
    print(f"[INFO] prompts={len(prompts)}")
    print(f"[INFO] participants={loop_cfg.participants}")
    print(f"[INFO] output_file={output_path}")

    for idx, prompt_item in enumerate(prompts, start=1):
        session = run_single_session(
            prompt_item=prompt_item,
            teachers=teachers,
            learning_cfg=learning_cfg,
            loop_cfg=loop_cfg,
            learning_client=learning_client,
        )
        append_jsonl(output_path, [session])
        print(
            f"[OK] prompt={idx}/{len(prompts)} "
            f"prompt_id={session['prompt_id']} "
            f"session_id={session['session_id']} "
            f"turns={len(session['turns'])}"
        )

    print("[INFO] run_conversation_loop finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
