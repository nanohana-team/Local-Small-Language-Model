from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

from src.apps.run_minimal_chat import (
    SurfaceNormalizer,
    build_dialogue_state_after,
    build_episode_actions,
    build_slot_trace,
    build_tokens,
)
from src.core.logging.trace_logger import JsonlTraceLogger
from src.core.planner.intent_planner import plan_intent
from src.core.recall.semantic_recall import recall_semantics
from src.core.scoring.basic_scorer import choose_best_response
from src.core.schema import (
    DialogueState,
    EvaluationResult,
    LexiconContainer,
    RealizationCandidate,
    TraceLog,
    build_input_state,
    build_runtime_state_snapshot,
    dataclass_to_dict,
    new_episode_id,
    new_session_id,
    new_turn_id,
)
from src.core.slots.slot_filler import fill_slots
from src.core.surface.surface_realizer import realize_surface
from src.training.external_evaluator import BaseExternalEvaluator, build_external_evaluator
from src.training.policy_memory import PolicyMemoryStore
from src.training.reward_aggregator import RewardAggregator, RewardAggregatorConfig
from src.training.target_generator import BaseTargetGenerator, GeneratedTarget, build_target_generator
from src.training.teacher_guidance import TeacherGuidanceConfig, TeacherGuidedReranker

LOGGER = logging.getLogger(__name__)
JST = ZoneInfo('Asia/Tokyo')


@dataclass(slots=True)
class LearningRuntimeConfig:
    trace_dir: str = 'runtime/traces'
    dataset_dir: str = 'runtime/datasets'
    save_trace: bool = True
    save_dataset: bool = True
    trace_latest_name: str = 'latest_trace.jsonl'
    dataset_latest_name: str = 'latest_dataset.jsonl'
    policy_memory_path: str = 'runtime/policy_memory.json'
    use_policy_memory: bool = True
    policy_memory_limit: int = 4
    teacher_target_weight: float = 0.35


@dataclass(slots=True)
class LearningEpisodeResult:
    session_id: str
    turn_id: str
    episode_id: str
    response_text: str
    reward_total: float
    trace: TraceLog
    evaluation: List[EvaluationResult]
    generated_target: GeneratedTarget
    trace_path: Optional[Path] = None
    dataset_path: Optional[Path] = None
    next_dialogue_state: Optional[DialogueState] = None


class JsonlDatasetWriter:
    def __init__(self, dataset_dir: str | Path, latest_name: str = 'latest_dataset.jsonl') -> None:
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.dataset_dir / latest_name

    def append(self, record: Mapping[str, Any]) -> Path:
        with self.latest_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(record), ensure_ascii=False, separators=(',', ':')))
            f.write('\n')
        return self.latest_path


def build_learning_dataset_record(trace: TraceLog, generated_target: GeneratedTarget) -> Dict[str, Any]:
    used_slots = dict(trace.response.used_slots) if trace.response else {}
    return {
        'record_type': 'training_sample',
        'timestamp': trace.timestamp,
        'session_id': trace.session_id,
        'turn_id': trace.turn_id,
        'episode_id': trace.episode_id,
        'input': trace.input_state.raw_text,
        'tokens': list(trace.input_state.normalized_tokens or trace.input_state.tokens),
        'intent': trace.intent_plan.intent,
        'slots': used_slots,
        'target': generated_target.text,
        'target_source': generated_target.source,
        'target_model': generated_target.model,
        'chosen_policy': trace.response.policy if trace.response else 'hold',
        'final_output': trace.response.text if trace.response else '',
        'reward': dataclass_to_dict(trace.reward),
        'evaluation': [dataclass_to_dict(item) for item in trace.evaluation],
        'teacher_guidance': dict(trace.debug.get('teacher_guidance', {}) or {}),
        'policy_memory': list(trace.debug.get('policy_memory_matches', []) or []),
    }


def _filled_slot_strings(trace_slots) -> Dict[str, str]:
    return {
        str(name): str(value.value)
        for name, value in trace_slots.values.items()
        if str(value.value).strip()
    }


def _merge_candidates(
    base_candidates: Sequence[RealizationCandidate],
    memory_candidates: Sequence[RealizationCandidate],
) -> List[RealizationCandidate]:
    merged: List[RealizationCandidate] = []
    seen: set[str] = set()
    for item in list(base_candidates) + list(memory_candidates):
        normalized = str(item.text).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged.append(item)
    return merged


def run_learning_episode(
    *,
    raw_text: str,
    lexicon: LexiconContainer,
    normalizer: SurfaceNormalizer,
    runtime_config: LearningRuntimeConfig | None = None,
    external_evaluator: BaseExternalEvaluator | None = None,
    reward_aggregator: RewardAggregator | None = None,
    target_generator: BaseTargetGenerator | None = None,
    explicit_words: Optional[Sequence[str]] = None,
    dialogue_state: Optional[DialogueState] = None,
    session_id: str = '',
    policy_memory: PolicyMemoryStore | None = None,
    teacher_reranker: TeacherGuidedReranker | None = None,
) -> LearningEpisodeResult:
    runtime_config = runtime_config or LearningRuntimeConfig()
    external_evaluator = external_evaluator or build_external_evaluator('llm')
    reward_aggregator = reward_aggregator or RewardAggregator(RewardAggregatorConfig())
    target_generator = target_generator or build_target_generator('llm')
    teacher_reranker = teacher_reranker or TeacherGuidedReranker(
        TeacherGuidanceConfig(target_weight=runtime_config.teacher_target_weight)
    )

    if runtime_config.use_policy_memory and policy_memory is None:
        policy_memory = PolicyMemoryStore(runtime_config.policy_memory_path, autoload=True)

    session_id = session_id or new_session_id('learnsess')
    turn_id = new_turn_id('learnturn')
    episode_id = new_episode_id('learnep')

    tokens = build_tokens(
        raw_text=raw_text,
        explicit_words=explicit_words,
        normalizer=normalizer,
    )
    if not tokens:
        raise ValueError('No tokens could be built from the given learning input.')

    input_state = build_input_state(
        raw_text=raw_text,
        tokens=tokens,
        normalized_tokens=tokens,
        session_id=session_id,
        turn_id=turn_id,
        timestamp=datetime.now(JST).isoformat(timespec='seconds'),
    )

    dialogue_state_current = deepcopy(dialogue_state) if dialogue_state is not None else DialogueState()
    dialogue_state_before = deepcopy(dialogue_state_current)

    LOGGER.info(
        'learning_episode.start session_id=%s turn_id=%s episode_id=%s raw_text=%s',
        session_id,
        turn_id,
        episode_id,
        raw_text,
    )

    intent_plan = plan_intent(input_state=input_state, dialogue_state=dialogue_state_current)
    recall_result = recall_semantics(
        input_state=input_state,
        lexicon=lexicon,
        dialogue_state=dialogue_state_current,
        intent_plan=intent_plan,
    )
    filled_slots = fill_slots(
        input_state=input_state,
        recall_result=recall_result,
        lexicon=lexicon,
        intent_plan=intent_plan,
        dialogue_state=dialogue_state_current,
    )
    surface_plan, base_candidates = realize_surface(
        filled_slots=filled_slots,
        intent_plan=intent_plan,
        lexicon=lexicon,
    )

    policy_memory_matches: List[Dict[str, object]] = []
    memory_candidates: List[RealizationCandidate] = []
    if runtime_config.use_policy_memory and policy_memory is not None:
        memory_candidates, policy_memory_matches = policy_memory.suggest(
            intent_plan=intent_plan,
            filled_slots=filled_slots,
            existing_texts=[item.text for item in base_candidates],
            limit=runtime_config.policy_memory_limit,
        )
        if memory_candidates:
            LOGGER.info(
                'learning_episode.policy_memory_augmented episode_id=%s count=%s',
                episode_id,
                len(memory_candidates),
            )

    merged_candidates = _merge_candidates(base_candidates, memory_candidates)

    base_response, scored_candidates = choose_best_response(
        input_state=input_state,
        intent_plan=intent_plan,
        filled_slots=filled_slots,
        candidates=merged_candidates,
    )

    used_slots_for_target = _filled_slot_strings(filled_slots)
    generated_target = target_generator.generate(
        user_input=raw_text,
        intent=intent_plan.intent,
        slots=used_slots_for_target,
        context={
            'input_tokens': list(input_state.normalized_tokens or input_state.tokens),
            'candidate_count': len(scored_candidates),
        },
    )

    teacher_guidance = teacher_reranker.rerank(
        candidates=scored_candidates,
        target_text=generated_target.text,
    )

    response = base_response
    if teacher_guidance.overridden and 0 <= teacher_guidance.selected_index < len(scored_candidates):
        selected_candidate = scored_candidates[teacher_guidance.selected_index]
        response, _ = choose_best_response(
            input_state=input_state,
            intent_plan=intent_plan,
            filled_slots=filled_slots,
            candidates=[selected_candidate],
        )
        LOGGER.info(
            'learning_episode.teacher_override episode_id=%s old=%s new=%s',
            episode_id,
            base_response.text,
            response.text,
        )

    evaluation_context = {
        'intent': intent_plan.intent,
        'required_slots': list(intent_plan.required_slots),
        'optional_slots': list(intent_plan.optional_slots),
        'used_slots': dict(response.used_slots),
        'input_tokens': list(input_state.normalized_tokens or input_state.tokens),
        'candidate_count': len(scored_candidates),
        'target_text': generated_target.text,
        'target_source': generated_target.source,
        'target_model': generated_target.model,
        'teacher_guidance': dataclass_to_dict(teacher_guidance),
    }
    evaluation = [
        external_evaluator.evaluate(
            user_input=raw_text,
            final_response=response.text,
            context=evaluation_context,
        )
    ]

    reward = reward_aggregator.aggregate(response=response, evaluation=evaluation)
    slot_trace = build_slot_trace(filled_slots)
    actions = build_episode_actions(
        intent_plan=intent_plan,
        recall_result=recall_result,
        filled_slots=filled_slots,
        surface_plan=surface_plan,
        scored_candidates=scored_candidates,
        response=response,
    )
    dialogue_state_after = build_dialogue_state_after(
        dialogue_state=dialogue_state_before,
        intent_plan=intent_plan,
        filled_slots=filled_slots,
        response=response,
    )

    policy_memory_path: Optional[Path] = None
    if runtime_config.use_policy_memory and policy_memory is not None:
        learned_slots = dict(response.used_slots) or used_slots_for_target
        policy_memory.remember(
            intent=intent_plan.intent,
            slots=learned_slots,
            text=generated_target.text,
            reward_total=max(reward.total, reward.external.total),
            internal_score=reward.internal.total,
            external_score=reward.external.total,
            source='teacher_target',
            template_id='teacher_target',
        )
        policy_memory.remember(
            intent=intent_plan.intent,
            slots=learned_slots,
            text=response.text,
            reward_total=reward.total,
            internal_score=reward.internal.total,
            external_score=reward.external.total,
            source='selected_response',
            template_id=response.chosen_candidate.template_id if response.chosen_candidate else '',
        )
        policy_memory_path = policy_memory.save()
        LOGGER.info('learning_episode.policy_memory_saved path=%s episode_id=%s', policy_memory_path, episode_id)

    trace = TraceLog(
        session_id=session_id,
        turn_id=turn_id,
        episode_id=episode_id,
        timestamp=datetime.now(JST).isoformat(timespec='seconds'),
        input_state=input_state,
        dialogue_state=dialogue_state_before,
        dialogue_state_after=dialogue_state_after,
        state_before=build_runtime_state_snapshot(input_state, dialogue_state_before),
        state_after=build_runtime_state_snapshot(input_state, dialogue_state_after),
        intent_plan=intent_plan,
        recall_result=recall_result,
        filled_slots=filled_slots,
        surface_plan=surface_plan,
        candidates=scored_candidates,
        actions=actions,
        slot_trace=slot_trace,
        response=response,
        reward=reward,
        evaluation=evaluation,
        debug={
            'mode': 'learning',
            'raw_text': raw_text,
            'tokens': tokens,
            'response_score': dataclass_to_dict(response.score),
            'reward': dataclass_to_dict(reward),
            'slot_trace': dataclass_to_dict(slot_trace),
            'evaluation_context': evaluation_context,
            'teacher_target': dataclass_to_dict(generated_target),
            'teacher_guidance': dataclass_to_dict(teacher_guidance),
            'base_response_text': base_response.text,
            'policy_memory_matches': policy_memory_matches,
            'policy_memory_path': str(policy_memory_path) if policy_memory_path else '',
        },
    )

    trace_path: Optional[Path] = None
    dataset_path: Optional[Path] = None

    if runtime_config.save_trace:
        trace_logger = JsonlTraceLogger(
            log_dir=runtime_config.trace_dir,
            latest_name=runtime_config.trace_latest_name,
            rotate_on_start=False,
        )
        trace_path = trace_logger.append_episode_trace(trace)
        LOGGER.info('learning_episode.trace_saved path=%s episode_id=%s', trace_path, episode_id)

    if runtime_config.save_dataset:
        dataset_writer = JsonlDatasetWriter(
            dataset_dir=runtime_config.dataset_dir,
            latest_name=runtime_config.dataset_latest_name,
        )
        dataset_path = dataset_writer.append(build_learning_dataset_record(trace, generated_target))
        LOGGER.info('learning_episode.dataset_saved path=%s episode_id=%s', dataset_path, episode_id)

    LOGGER.info(
        'learning_episode.done episode_id=%s total=%.4f internal=%.4f external=%.4f target=%s response=%s',
        episode_id,
        reward.total,
        reward.internal.total,
        reward.external.total,
        generated_target.text,
        response.text,
    )

    return LearningEpisodeResult(
        session_id=session_id,
        turn_id=turn_id,
        episode_id=episode_id,
        response_text=response.text,
        reward_total=reward.total,
        trace=trace,
        evaluation=evaluation,
        generated_target=generated_target,
        trace_path=trace_path,
        dataset_path=dataset_path,
        next_dialogue_state=dialogue_state_after,
    )
