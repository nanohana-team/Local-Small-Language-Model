from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping


@dataclass
class TeacherTurnRequest:
    """Normalized request envelope for external teacher calls.

    This keeps the teacher-facing payload stable even when the internal turn trace
    grows. The full trace can still be stored elsewhere, but the teacher should
    receive a compact, intentional input shape.
    """

    input_text: str
    response: str
    plan_summary: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    scores: Dict[str, Any] = field(default_factory=dict)
    reward: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_turn_payload(cls, payload: Mapping[str, Any]) -> "TeacherTurnRequest":
        plan = payload.get("plan") if isinstance(payload.get("plan"), Mapping) else {}
        filled_slots = payload.get("filled_slots") if isinstance(payload.get("filled_slots"), Mapping) else {}
        accepted_relations = payload.get("accepted_relations") if isinstance(payload.get("accepted_relations"), list) else []
        scores = payload.get("scores") if isinstance(payload.get("scores"), Mapping) else {}
        reward = payload.get("reward") if isinstance(payload.get("reward"), Mapping) else {}

        plan_summary = {
            "intent": plan.get("intent"),
            "response_mode": plan.get("response_mode"),
            "required_slots": list(plan.get("required_slots") or []),
            "relation_type_priority": list(plan.get("relation_type_priority") or []),
            "needs_clarification": bool(plan.get("needs_clarification", False)),
        }
        constraints = {
            "required_slots": list(plan.get("required_slots") or []),
            "filled_slot_names": sorted(str(key) for key in filled_slots.keys()),
            "accepted_relation_count": len(accepted_relations),
        }
        return cls(
            input_text=str(payload.get("input") or ""),
            response=str(payload.get("response") or ""),
            plan_summary=plan_summary,
            constraints=constraints,
            scores=dict(scores),
            reward=dict(reward),
        )

    def to_payload(self) -> Dict[str, Any]:
        return asdict(self)
