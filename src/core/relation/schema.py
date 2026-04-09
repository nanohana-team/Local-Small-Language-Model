from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping

from src.core.io.lsd_lexicon import (
    RELATION_DIRECTIONS,
    RELATION_LAYERS,
    RELATION_TYPE_RULES,
    RELATION_USAGE_STAGES,
)


@dataclass(frozen=True)
class RelationRule:
    """Canonical rule for a relation type."""

    type: str
    layer: str
    direction: str
    usage_stage: tuple[str, ...]
    inverse_type: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_relation_rule(relation_type: str) -> RelationRule | None:
    raw_rule = RELATION_TYPE_RULES.get(str(relation_type))
    if raw_rule is None:
        return None
    return RelationRule(
        type=str(relation_type),
        layer=str(raw_rule.get("layer", "semantic")),
        direction=str(raw_rule.get("direction", "outbound")),
        usage_stage=tuple(str(v) for v in raw_rule.get("usage_stage", [])),
        inverse_type=str(raw_rule["inverse_type"]) if raw_rule.get("inverse_type") is not None else None,
    )


def canonicalize_relation(relation: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize a relation mapping into the minimal formal contract.

    This function deliberately does not try to infer a missing target.
    It only fills stable defaults that are safe to derive from the taxonomy.
    """

    relation_type = str(relation.get("type", "")).strip()
    target = str(relation.get("target", "")).strip()
    rule = get_relation_rule(relation_type)

    direction = str(relation.get("direction") or (rule.direction if rule else "outbound"))
    if direction not in RELATION_DIRECTIONS:
        raise ValueError(f"Unsupported relation direction: {direction!r}")

    layer = str(relation.get("layer") or (rule.layer if rule else "semantic"))
    if layer not in RELATION_LAYERS:
        raise ValueError(f"Unsupported relation layer: {layer!r}")

    usage_stage_raw = relation.get("usage_stage") or (list(rule.usage_stage) if rule else ["divergence"])
    usage_stage = [str(v) for v in usage_stage_raw]
    unknown_stages = [stage for stage in usage_stage if stage not in RELATION_USAGE_STAGES]
    if unknown_stages:
        raise ValueError(f"Unsupported relation usage stages: {unknown_stages!r}")

    canonical: Dict[str, Any] = {
        "type": relation_type,
        "target": target,
        "weight": float(relation.get("weight", 1.0)),
        "direction": direction,
        "layer": layer,
        "usage_stage": usage_stage,
        "confidence": float(relation.get("confidence", relation.get("weight", 1.0))),
    }

    inverse_type = relation.get("inverse_type")
    if inverse_type is None and rule is not None:
        inverse_type = rule.inverse_type
    if inverse_type is not None:
        canonical["inverse_type"] = str(inverse_type)

    for optional_key in ("axes", "constraints", "evidence", "meta"):
        if optional_key in relation:
            canonical[optional_key] = relation[optional_key]

    return canonical


__all__ = [
    "RELATION_DIRECTIONS",
    "RELATION_LAYERS",
    "RELATION_TYPE_RULES",
    "RELATION_USAGE_STAGES",
    "RelationRule",
    "canonicalize_relation",
    "get_relation_rule",
]
