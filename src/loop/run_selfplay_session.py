from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List

try:
    from src.inference.local_student_client import LearningConfig, LocalStudentClient
except Exception:
    try:
        from src.tools.local_student_client import LearningConfig, LocalStudentClient
    except Exception:
        from local_student_client import LearningConfig, LocalStudentClient

try:
    from src.inference.teacher_client import TeacherConfig, generate_teacher_reply
except Exception:
    try:
        from src.tools.teacher_client import TeacherConfig, generate_teacher_reply
    except Exception:
        from teacher_client import TeacherConfig, generate_teacher_reply

try:
    from src.loop.session_types import SessionRecord, TurnRecord, utc_now_iso
except Exception:
    try:
        from src.tools.session_types import SessionRecord, TurnRecord, utc_now_iso
    except Exception:
        from session_types import SessionRecord, TurnRecord, utc_now_iso


@dataclass
class ConversationLoopConfig:
    max_rounds: int
    participants: List[str]
    history_strategy: str = "full_turns"
    turn_sleep_sec: float = 0.0
    stop_on_empty_response: bool = True
    include_input_as_first_user_turn: bool = True
    max_context_messages: int = 32

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ConversationLoopConfig":
        return ConversationLoopConfig(
            max_rounds=int(d.get("max_rounds", 4)),
            participants=[str(x) for x in d.get("participants", [])],
            history_strategy=str(d.get("history_strategy", "full_turns")),
            turn_sleep_sec=float(d.get("turn_sleep_sec", 0.0)),
            stop_on_empty_response=bool(d.get("stop_on_empty_response", True)),
            include_input_as_first_user_turn=bool(d.get("include_input_as_first_user_turn", True)),
            max_context_messages=int(d.get("max_context_messages", 32)),
        )


def build_messages_for_next_turn(session: SessionRecord, max_context_messages: int) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "user", "content": session.input_text}]
    for turn in session.turns:
        messages.append({"role": turn.role, "content": turn.text})
    if max_context_messages > 0:
        messages = messages[-max_context_messages:]
    return messages


def run_single_session(
    prompt_item: Dict[str, Any],
    teachers: Dict[str, TeacherConfig],
    learning_cfg: LearningConfig,
    loop_cfg: ConversationLoopConfig,
    learning_client: LocalStudentClient,
) -> Dict[str, Any]:
    prompt_id = str(prompt_item.get("id") or f"prompt_{uuid.uuid4().hex[:8]}")
    input_text = str(prompt_item.get("input", "")).strip()
    if not input_text:
        raise ValueError(f"Prompt item '{prompt_id}' does not contain non-empty 'input'.")

    session = SessionRecord(
        session_id=str(uuid.uuid4()),
        prompt_id=prompt_id,
        input_text=input_text,
        created_at=utc_now_iso(),
        meta={
            "max_rounds": loop_cfg.max_rounds,
            "participants": loop_cfg.participants,
            "source_meta": prompt_item.get("meta", {}),
        },
    )

    total_turns = loop_cfg.max_rounds * len(loop_cfg.participants)
    for turn_index in range(total_turns):
        speaker_name = loop_cfg.participants[turn_index % len(loop_cfg.participants)]
        messages = build_messages_for_next_turn(session, loop_cfg.max_context_messages)

        started = time.perf_counter()
        if speaker_name == learning_cfg.name:
            text = learning_client.generate(messages)
            model_name = learning_cfg.model
        else:
            teacher = teachers[speaker_name]
            text = generate_teacher_reply(teacher, messages)
            model_name = teacher.model
        elapsed = round(time.perf_counter() - started, 4)

        text = (text or "").strip()
        if not text and loop_cfg.stop_on_empty_response:
            print(f"[WARN] empty response from {speaker_name}; stopping session {session.session_id}")
            break

        session.turns.append(
            TurnRecord(
                turn_index=turn_index,
                speaker=speaker_name,
                model=model_name,
                text=text,
                latency_sec=elapsed,
                role="assistant",
                meta={},
            )
        )

        if loop_cfg.turn_sleep_sec > 0:
            time.sleep(loop_cfg.turn_sleep_sec)

    return session.to_dict()
