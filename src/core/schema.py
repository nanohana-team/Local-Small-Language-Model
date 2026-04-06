from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from zoneinfo import ZoneInfo


JST = ZoneInfo("Asia/Tokyo")

IntentType = Literal[
    "respond",
    "empathy",
    "question",
    "confirm",
    "explain",
    "unknown",
]

PolicyType = Literal[
    "answer",
    "ask_back",
    "agree",
    "hold",
    "clarify",
]

RecallSourceType = Literal[
    "input",
    "relation",
    "axis",
    "fallback",
]

ActionStageType = Literal[
    "intent",
    "recall",
    "slot",
    "surface",
    "evaluation",
    "unknown",
]

SlotName = Literal[
    "actor",
    "target",
    "recipient",
    "location",
    "time",
    "cause",
    "state",
    "predicate",
    "topic",
    "manner",
]

DEFAULT_SEMANTIC_AXES: List[str] = [
    "valence",
    "arousal",
    "abstractness",
    "sociality",
    "temporality",
    "agency",
    "causality",
    "certainty",
    "deixis",
    "discourse_force",
]


@dataclass(slots=True)
class AxisVector:
    valence: float = 0.0
    arousal: float = 0.0
    abstractness: float = 0.0
    sociality: float = 0.0
    temporality: float = 0.0
    agency: float = 0.0
    causality: float = 0.0
    certainty: float = 0.0
    deixis: float = 0.0
    discourse_force: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "valence": float(self.valence),
            "arousal": float(self.arousal),
            "abstractness": float(self.abstractness),
            "sociality": float(self.sociality),
            "temporality": float(self.temporality),
            "agency": float(self.agency),
            "causality": float(self.causality),
            "certainty": float(self.certainty),
            "deixis": float(self.deixis),
            "discourse_force": float(self.discourse_force),
        }

    def ordered_values(self) -> List[float]:
        data = self.to_dict()
        return [float(data[axis]) for axis in DEFAULT_SEMANTIC_AXES]

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AxisVector":
        if not data:
            return cls()
        return cls(
            valence=float(data.get("valence", 0.0)),
            arousal=float(data.get("arousal", 0.0)),
            abstractness=float(data.get("abstractness", 0.0)),
            sociality=float(data.get("sociality", 0.0)),
            temporality=float(data.get("temporality", 0.0)),
            agency=float(data.get("agency", 0.0)),
            causality=float(data.get("causality", 0.0)),
            certainty=float(data.get("certainty", 0.0)),
            deixis=float(data.get("deixis", 0.0)),
            discourse_force=float(data.get("discourse_force", 0.0)),
        )


@dataclass(slots=True)
class GrammarConstraints:
    pos: str
    sub_pos: str = ""
    can_start: bool = False
    can_end: bool = False
    independent: bool = True
    content_word: bool = True
    function_word: bool = False
    requires_prev: List[str] = field(default_factory=list)
    requires_next: List[str] = field(default_factory=list)
    forbid_prev: List[str] = field(default_factory=list)
    forbid_next: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "GrammarConstraints":
        if not data:
            return cls(pos="unknown")
        return cls(
            pos=str(data.get("pos", "unknown")),
            sub_pos=str(data.get("sub_pos", "")),
            can_start=bool(data.get("can_start", False)),
            can_end=bool(data.get("can_end", False)),
            independent=bool(data.get("independent", True)),
            content_word=bool(data.get("content_word", True)),
            function_word=bool(data.get("function_word", False)),
            requires_prev=[str(x) for x in data.get("requires_prev", [])],
            requires_next=[str(x) for x in data.get("requires_next", [])],
            forbid_prev=[str(x) for x in data.get("forbid_prev", [])],
            forbid_next=[str(x) for x in data.get("forbid_next", [])],
        )


@dataclass(slots=True)
class SlotConstraint:
    name: str
    required: bool = False
    allowed_pos: List[str] = field(default_factory=list)
    semantic_hint: List[str] = field(default_factory=list)
    note: str = ""

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "SlotConstraint":
        if not data:
            return cls(name="")
        return cls(
            name=str(data.get("name", "")),
            required=bool(data.get("required", False)),
            allowed_pos=[str(x) for x in data.get("allowed_pos", [])],
            semantic_hint=[str(x) for x in data.get("semantic_hint", [])],
            note=str(data.get("note", "")),
        )


@dataclass(slots=True)
class RelationEdge:
    relation: str
    target: str
    weight: float = 1.0
    bidirectional: bool = False
    note: str = ""

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "RelationEdge":
        if not data:
            return cls(relation="", target="")
        return cls(
            relation=str(data.get("relation", "")),
            target=str(data.get("target", "")),
            weight=float(data.get("weight", 1.0)),
            bidirectional=bool(data.get("bidirectional", False)),
            note=str(data.get("note", "")),
        )


@dataclass(slots=True)
class SurfaceForm:
    form: str = ""
    surface: str = ""
    tokens: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "SurfaceForm":
        if not data:
            return cls()
        surface = str(data.get("surface", ""))
        tokens = [str(x) for x in data.get("tokens", [])]
        if not tokens and surface:
            tokens = [surface]
        return cls(
            form=str(data.get("form", "")),
            surface=surface,
            tokens=tokens,
        )


@dataclass(slots=True)
class LexiconEntry:
    word: str
    category: str
    hierarchy: List[str] = field(default_factory=list)
    vector: AxisVector = field(default_factory=AxisVector)
    grammar: GrammarConstraints = field(
        default_factory=lambda: GrammarConstraints(pos="unknown")
    )
    slots: List[SlotConstraint] = field(default_factory=list)
    relations: List[RelationEdge] = field(default_factory=list)
    frequency: float = 0.0
    style_tags: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    entry_type: str = "surface"
    lemma: str = ""
    stem_id: str = ""
    conj_class: str = ""
    effect: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    surface_forms: List[SurfaceForm] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LexiconEntry":
        return cls(
            word=str(data.get("word", "")),
            category=str(data.get("category", "")),
            hierarchy=[str(x) for x in data.get("hierarchy", [])],
            vector=AxisVector.from_dict(data.get("vector")),
            grammar=GrammarConstraints.from_dict(data.get("grammar")),
            slots=[SlotConstraint.from_dict(x) for x in data.get("slots", [])],
            relations=[RelationEdge.from_dict(x) for x in data.get("relations", [])],
            frequency=float(data.get("frequency", 0.0)),
            style_tags=[str(x) for x in data.get("style_tags", [])],
            meta=dict(data.get("meta", {})),
            entry_type=str(data.get("type", data.get("entry_type", "surface"))),
            lemma=str(data.get("lemma", data.get("word", ""))),
            stem_id=str(data.get("stem_id", "")),
            conj_class=str(data.get("conj_class", "")),
            effect=dict(data.get("effect", {})),
            aliases=[str(x) for x in data.get("aliases", [])],
            surface_forms=[SurfaceForm.from_dict(x) for x in data.get("surface_forms", [])],
        )

    def get_surface(self, preferred_form: str = "plain") -> str:
        if self.surface_forms:
            for item in self.surface_forms:
                if item.form == preferred_form and item.surface:
                    return item.surface
            for item in self.surface_forms:
                if item.form == "plain" and item.surface:
                    return item.surface
            for item in self.surface_forms:
                if item.surface:
                    return item.surface
        if self.lemma:
            return self.lemma
        return self.word


@dataclass(slots=True)
class LexiconMeta:
    semantic_axes: List[str] = field(
        default_factory=lambda: list(DEFAULT_SEMANTIC_AXES)
    )
    entry_count: int = 0
    version: str = "v3"

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "LexiconMeta":
        if not data:
            return cls()
        return cls(
            semantic_axes=[str(x) for x in data.get("semantic_axes", DEFAULT_SEMANTIC_AXES)],
            entry_count=int(data.get("entry_count", 0)),
            version=str(data.get("version", "v3")),
        )


@dataclass(slots=True)
class LexiconIndexes:
    by_pos: Dict[str, List[str]] = field(default_factory=dict)
    can_start: List[str] = field(default_factory=list)
    can_end: List[str] = field(default_factory=list)
    content_words: List[str] = field(default_factory=list)
    function_words: List[str] = field(default_factory=list)
    entry_path: Dict[str, List[str]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "LexiconIndexes":
        if not data:
            return cls()
        return cls(
            by_pos={
                str(k): [str(x) for x in v]
                for k, v in dict(data.get("by_pos", {})).items()
            },
            can_start=[str(x) for x in data.get("can_start", [])],
            can_end=[str(x) for x in data.get("can_end", [])],
            content_words=[str(x) for x in data.get("content_words", data.get("content_word", []))],
            function_words=[str(x) for x in data.get("function_words", data.get("function_word", []))],
            entry_path={
                str(k): [str(x) for x in v]
                for k, v in dict(data.get("entry_path", {})).items()
            },
        )


@dataclass(slots=True)
class LexiconContainer:
    meta: LexiconMeta = field(default_factory=LexiconMeta)
    entries: Dict[str, LexiconEntry] = field(default_factory=dict)
    indexes: LexiconIndexes = field(default_factory=LexiconIndexes)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "LexiconContainer":
        if not data:
            return cls()
        entries_src = dict(data.get("entries", {}))
        return cls(
            meta=LexiconMeta.from_dict(data.get("meta")),
            entries={
                str(word): LexiconEntry.from_dict(entry)
                for word, entry in entries_src.items()
            },
            indexes=LexiconIndexes.from_dict(data.get("indexes")),
        )


@dataclass(slots=True)
class InputState:
    raw_text: str
    tokens: List[str]
    normalized_tokens: List[str] = field(default_factory=list)
    timestamp: str = ""
    session_id: str = ""
    turn_id: str = ""


@dataclass(slots=True)
class DialogueState:
    current_topic: str = ""
    last_subject: str = ""
    last_object: str = ""
    referents: Dict[str, str] = field(default_factory=dict)
    context_vector: AxisVector = field(default_factory=AxisVector)
    inferred_intent_history: List[str] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IntentPlan:
    intent: IntentType = "unknown"
    confidence: float = 0.0
    required_slots: List[str] = field(default_factory=list)
    optional_slots: List[str] = field(default_factory=list)
    response_policy_hint: PolicyType = "hold"
    note: str = ""


@dataclass(slots=True)
class RecallCandidate:
    word: str
    score: float
    source: RecallSourceType = "fallback"
    relation_path: List[str] = field(default_factory=list)
    axis_distance: float = 0.0
    grammar_ok: bool = True
    note: str = ""


@dataclass(slots=True)
class RecallResult:
    seeds: List[str] = field(default_factory=list)
    candidates: List[RecallCandidate] = field(default_factory=list)


@dataclass(slots=True)
class SlotValue:
    slot_name: str
    value: str
    confidence: float = 0.0
    source_candidate: str = ""
    inferred: bool = False
    note: str = ""


@dataclass(slots=True)
class SlotFrame:
    predicate: str = ""
    predicate_type: str = ""
    constraints: List[SlotConstraint] = field(default_factory=list)


@dataclass(slots=True)
class FilledSlots:
    frame: SlotFrame = field(default_factory=SlotFrame)
    values: Dict[str, SlotValue] = field(default_factory=dict)
    missing_required: List[str] = field(default_factory=list)
    optional_unfilled: List[str] = field(default_factory=list)
    consistency_score: float = 0.0


@dataclass(slots=True)
class SurfacePlan:
    template_id: str = ""
    style: str = "neutral"
    politeness: str = "plain"
    sentence_count: int = 1
    order: List[str] = field(default_factory=list)
    auxiliaries: List[str] = field(default_factory=list)
    note: str = ""


@dataclass(slots=True)
class RealizationCandidate:
    text: str
    token_sequence: List[str] = field(default_factory=list)
    template_id: str = ""
    grammar_violations: List[str] = field(default_factory=list)
    slot_coverage: float = 0.0
    semantic_score: float = 0.0
    final_score: float = 0.0


@dataclass(slots=True)
class ScoreBreakdown:
    semantic_consistency: float = 0.0
    slot_fitness: float = 0.0
    grammar_fitness: float = 0.0
    input_retention: float = 0.0
    policy_fitness: float = 0.0
    total: float = 0.0
    reasons: List[str] = field(default_factory=list)


@dataclass(slots=True)
class ActionCandidateSnapshot:
    key: str = ""
    label: str = ""
    score: float = 0.0
    rank: int = 0
    source: str = ""
    kept: bool = False
    dropped: bool = False
    drop_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EpisodeAction:
    stage: ActionStageType = "unknown"
    action_type: str = ""
    selected: Any = None
    candidates: List[ActionCandidateSnapshot] = field(default_factory=list)
    confidence: float = 0.0
    note: str = ""
    candidate_count: int = 0
    selected_count: int = 0
    dropped_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SlotTraceItem:
    slot_name: str
    expected: bool = False
    required: bool = False
    filled: bool = False
    value: str = ""
    confidence: float = 0.0
    source_candidate: str = ""
    inferred: bool = False
    note: str = ""


@dataclass(slots=True)
class SlotTrace:
    predicate: str = ""
    predicate_type: str = ""
    frame_constraints: List[str] = field(default_factory=list)
    all_slots: List[str] = field(default_factory=list)
    filled_slots: List[SlotTraceItem] = field(default_factory=list)
    missing_required: List[str] = field(default_factory=list)
    optional_unfilled: List[str] = field(default_factory=list)
    consistency_score: float = 0.0


@dataclass(slots=True)
class InternalRewardBreakdown:
    semantic: float = 0.0
    slot: float = 0.0
    grammar: float = 0.0
    retention: float = 0.0
    policy: float = 0.0
    total: float = 0.0
    reasons: List[str] = field(default_factory=list)


@dataclass(slots=True)
class ExternalRewardComponent:
    evaluator_name: str = ""
    score: float = 0.0
    weight: float = 1.0
    weighted_score: float = 0.0
    label: str = ""
    feedback: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExternalRewardBreakdown:
    components: List[ExternalRewardComponent] = field(default_factory=list)
    total: float = 0.0


@dataclass(slots=True)
class RewardBreakdown:
    internal: InternalRewardBreakdown = field(default_factory=InternalRewardBreakdown)
    external: ExternalRewardBreakdown = field(default_factory=ExternalRewardBreakdown)
    total: float = 0.0
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ResponseResult:
    text: str
    intent: IntentType = "unknown"
    policy: PolicyType = "hold"
    chosen_candidate: Optional[RealizationCandidate] = None
    score: ScoreBreakdown = field(default_factory=ScoreBreakdown)
    used_relations: List[str] = field(default_factory=list)
    used_slots: Dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationResult:
    evaluator_name: str
    score: float
    label: str = ""
    feedback: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TraceLog:
    session_id: str
    turn_id: str
    input_state: InputState
    dialogue_state: DialogueState
    intent_plan: IntentPlan
    recall_result: RecallResult
    filled_slots: FilledSlots
    surface_plan: SurfacePlan
    candidates: List[RealizationCandidate] = field(default_factory=list)
    response: Optional[ResponseResult] = None
    evaluation: List[EvaluationResult] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)

    episode_id: str = ""
    timestamp: str = ""
    record_type: str = "episode_trace"
    state_before: Dict[str, Any] = field(default_factory=dict)
    actions: List[EpisodeAction] = field(default_factory=list)
    slot_trace: SlotTrace = field(default_factory=SlotTrace)
    reward: RewardBreakdown = field(default_factory=RewardBreakdown)
    state_after: Dict[str, Any] = field(default_factory=dict)
    dialogue_state_after: Optional[DialogueState] = None


def dataclass_to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {k: dataclass_to_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [dataclass_to_dict(v) for v in value]
    if isinstance(value, tuple):
        return [dataclass_to_dict(v) for v in value]
    return value


def build_input_state(
    raw_text: str,
    tokens: List[str],
    normalized_tokens: Optional[List[str]] = None,
    session_id: str = "",
    turn_id: str = "",
    timestamp: Optional[str] = None,
) -> InputState:
    return InputState(
        raw_text=raw_text,
        tokens=list(tokens),
        normalized_tokens=list(normalized_tokens or tokens),
        timestamp=timestamp or datetime.now(JST).isoformat(timespec="seconds"),
        session_id=session_id,
        turn_id=turn_id,
    )


def build_dialogue_state_snapshot(dialogue_state: DialogueState) -> Dict[str, Any]:
    return {
        "current_topic": dialogue_state.current_topic,
        "last_subject": dialogue_state.last_subject,
        "last_object": dialogue_state.last_object,
        "referents": dict(dialogue_state.referents),
        "context_vector": dialogue_state.context_vector.to_dict(),
        "inferred_intent_history": list(dialogue_state.inferred_intent_history),
        "variables": dict(dialogue_state.variables),
    }


def build_runtime_state_snapshot(
    input_state: InputState,
    dialogue_state: DialogueState,
) -> Dict[str, Any]:
    return {
        "raw_text": input_state.raw_text,
        "tokens": list(input_state.tokens),
        "normalized_tokens": list(input_state.normalized_tokens),
        "dialogue_state": build_dialogue_state_snapshot(dialogue_state),
    }


def new_session_id(prefix: str = "sess") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def new_turn_id(prefix: str = "turn") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def new_episode_id(prefix: str = "ep") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"