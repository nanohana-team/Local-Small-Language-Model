"""Relation schema, validation, and indexes for LSLM v4."""

from .index import RelationIndex, build_relation_index
from .schema import RELATION_TYPE_RULES, RelationRule, canonicalize_relation, get_relation_rule
from .validator import validate_relation_graph

__all__ = [
    "RELATION_TYPE_RULES",
    "RelationIndex",
    "RelationRule",
    "build_relation_index",
    "canonicalize_relation",
    "get_relation_rule",
    "validate_relation_graph",
]
