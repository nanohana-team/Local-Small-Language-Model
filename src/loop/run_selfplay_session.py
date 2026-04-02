from __future__ import annotations

import json
import random
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from src.inference.local_student_client import LearningConfig, LocalStudentClient
from src.inference.teacher_client import TeacherConfig, generate_teacher_reply
from src.loop.session_types import SessionRecord, TurnRecord, utc_now_iso


@dataclass
class ConversationLoopConfig:
    max_rounds: int
    participants: List[str]
    history_strategy: str = "full_turns"
    turn_sleep_sec: float = 0.0
    stop_on_empty_response: bool = True
    include_input_as_first_user_turn: bool = True
    max_context_messages: int = 32
    random_order: bool = True  # ラウンドごとに発言順をシャッフル

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
            random_order=bool(d.get("random_order", True)),
        )


def build_messages_for_next_turn(
    session: SessionRecord,
    max_context_messages: int,
    include_input_as_first_user_turn: bool = True,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if include_input_as_first_user_turn:
        messages.append({"role": "user", "content": session.input_text})
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
            "random_order": loop_cfg.random_order,
            "source_meta": prompt_item.get("meta", {}),
        },
    )

    turn_index = 0
    stop = False

    for round_idx in range(loop_cfg.max_rounds):
        if stop:
            break

        # ラウンドごとに発言順を決定（random_order=True のときシャッフル）
        speakers_this_round = list(loop_cfg.participants)
        if loop_cfg.random_order:
            random.shuffle(speakers_this_round)

        print(f"  [Round {round_idx + 1}/{loop_cfg.max_rounds}] order={speakers_this_round}")

        for speaker_name in speakers_this_round:
            messages = build_messages_for_next_turn(
                session,
                loop_cfg.max_context_messages,
                loop_cfg.include_input_as_first_user_turn,
            )

            started = time.perf_counter()
            try:
                if speaker_name == learning_cfg.name:
                    text = learning_client.generate(messages)
                    model_name = learning_cfg.model
                else:
                    teacher = teachers[speaker_name]
                    text = generate_teacher_reply(teacher, messages)
                    model_name = teacher.model
            except Exception as exc:
                print(f"  [WARN] speaker={speaker_name} failed: {exc}")
                if loop_cfg.stop_on_empty_response:
                    stop = True
                    break
                continue

            elapsed = round(time.perf_counter() - started, 4)
            text = (text or "").strip()

            if not text and loop_cfg.stop_on_empty_response:
                print(f"  [STOP] empty response from speaker={speaker_name}")
                stop = True
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
            print(f"  [Turn {turn_index}] speaker={speaker_name} chars={len(text)} latency={elapsed}s")
            turn_index += 1

            if loop_cfg.turn_sleep_sec > 0:
                time.sleep(loop_cfg.turn_sleep_sec)

    return session.to_dict()
