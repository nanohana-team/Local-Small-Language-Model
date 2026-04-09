from __future__ import annotations

from typing import Any, Dict, Mapping

from .index import RelationIndex, build_relation_index
from .schema import RELATION_DIRECTIONS, RELATION_LAYERS, RELATION_USAGE_STAGES, get_relation_rule



def _append_issue(report: Dict[str, Any], bucket: str, message: str) -> None:
    report[bucket].append(str(message))



def validate_relation_graph(
    container: Mapping[str, Any],
    *,
    strict_schema: bool = False,
    strict_relations: bool = True,
    require_closed_relations: bool = True,
    prebuilt_index: RelationIndex | None = None,
) -> Dict[str, Any]:
    """Fast relation-focused validator for Phase 2 work.

    It intentionally validates the concept graph and its slot-frame references,
    instead of re-walking every lexical entry in the container. That keeps the
    check cheap enough to run at engine startup while still enforcing the
    core Phase 2 contract: schema shape, closed graph targets, direction rules,
    usage stages, and inverse consistency.
    """

    report: Dict[str, Any] = {"errors": [], "warnings": []}
    concepts = container.get("concepts")
    slot_frames = container.get("slot_frames")

    if not isinstance(concepts, Mapping):
        _append_issue(report, "errors", "top-level concepts must be a mapping")
        report["error_count"] = len(report["errors"])
        report["warning_count"] = len(report["warnings"])
        report["ok"] = False
        return report
    if strict_schema and not isinstance(slot_frames, Mapping):
        _append_issue(report, "errors", "strict schema mode requires top-level slot_frames mapping")
        report["error_count"] = len(report["errors"])
        report["warning_count"] = len(report["warnings"])
        report["ok"] = False
        return report
    if not isinstance(slot_frames, Mapping):
        slot_frames = {}

    index = prebuilt_index or build_relation_index(container)

    for concept_id, concept in index.concepts.items():
        label = concept.get("label")
        if strict_schema and (not isinstance(label, str) or not label.strip()):
            _append_issue(report, "errors", f"concept {concept_id!r} requires non-empty label in strict schema mode")

        default_slot_frame_id = concept.get("default_slot_frame_id")
        if default_slot_frame_id is not None and str(default_slot_frame_id) not in slot_frames:
            _append_issue(report, "errors", f"concept {concept_id!r} references missing slot frame {default_slot_frame_id!r}")

        relations = concept.get("relations", [])
        if strict_schema and not isinstance(relations, list):
            _append_issue(report, "errors", f"concept {concept_id!r} relations must be a list")
            continue
        if not isinstance(relations, list):
            continue

        for rel_index, relation in enumerate(relations):
            rel_label = f"concept {concept_id!r} relation[{rel_index}]"
            if not isinstance(relation, Mapping):
                _append_issue(report, "errors", f"{rel_label} must be a mapping")
                continue

            rel_type = str(relation.get("type", "")).strip()
            target = str(relation.get("target", "")).strip()
            if not rel_type:
                _append_issue(report, "errors", f"{rel_label} requires non-empty type")
                continue
            if not target:
                _append_issue(report, "errors", f"{rel_label} requires non-empty target")
                continue

            rule = get_relation_rule(rel_type)
            if rule is None:
                issue_bucket = "errors" if strict_relations else "warnings"
                _append_issue(report, issue_bucket, f"{rel_label} uses unknown relation type {rel_type!r}")
            direction = str(relation.get("direction") or (rule.direction if rule else "outbound"))
            if direction not in RELATION_DIRECTIONS:
                _append_issue(report, "errors", f"{rel_label} has unsupported direction {direction!r}")
            elif rule is not None and direction != rule.direction:
                _append_issue(report, "warnings", f"{rel_label} direction {direction!r} differs from canonical {rule.direction!r}")

            layer = str(relation.get("layer") or (rule.layer if rule else "semantic"))
            if layer not in RELATION_LAYERS:
                _append_issue(report, "errors", f"{rel_label} has unsupported layer {layer!r}")
            elif rule is not None and layer != rule.layer:
                _append_issue(report, "warnings", f"{rel_label} layer {layer!r} differs from canonical {rule.layer!r}")

            usage_stage = relation.get("usage_stage") or (list(rule.usage_stage) if rule else ["divergence"])
            if not isinstance(usage_stage, list) or not usage_stage:
                _append_issue(report, "errors", f"{rel_label} usage_stage must be a non-empty list")
            else:
                unknown_stages = [str(stage) for stage in usage_stage if str(stage) not in RELATION_USAGE_STAGES]
                if unknown_stages:
                    _append_issue(report, "errors", f"{rel_label} has unknown usage stages {unknown_stages!r}")

            if require_closed_relations and target not in index.concepts:
                _append_issue(report, "errors", f"{rel_label} points to missing concept {target!r}")
                continue

            if target in index.concepts:
                outbound_target_relations = index.get_outbound(target)
                if direction == "bidirectional":
                    mirrored = any(
                        str(candidate.get("target")) == concept_id and str(candidate.get("type")) == rel_type
                        for candidate in outbound_target_relations
                    )
                    if not mirrored:
                        _append_issue(report, "warnings", f"{rel_label} is bidirectional but mirror edge is missing")

                expected_inverse = relation.get("inverse_type") or (rule.inverse_type if rule else None)
                if expected_inverse:
                    has_inverse = any(
                        str(candidate.get("target")) == concept_id and str(candidate.get("type")) == str(expected_inverse)
                        for candidate in outbound_target_relations
                    )
                    if not has_inverse:
                        _append_issue(
                            report,
                            "warnings",
                            f"{rel_label} is missing inverse edge {expected_inverse!r} from {target!r}",
                        )

    report["error_count"] = len(report["errors"])
    report["warning_count"] = len(report["warnings"])
    report["ok"] = report["error_count"] == 0
    return report


__all__ = ["validate_relation_graph"]
