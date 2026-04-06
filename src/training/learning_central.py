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
    build_tokenization_result,
    build_tokens,
)
from src.core.logging.trace_logger import JsonlTraceLogger
from src.core.planner.intent_planner import IntentPlannerConfig, plan_intent
from src.core.recall.semantic_recall import SemanticRecallConfig, recall_semantics
from src.core.scoring.basic_scorer import BasicScorerConfig, choose_best_response
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
from src.core.slots.slot_filler import SlotFillerConfig, fill_slots
from src.core.surface.surface_realizer import SurfaceRealizerConfig, realize_surface
from src.training.action_bandit import ActionBanditConfig, ActionBanditStore
from src.training.external_evaluator import BaseExternalEvaluator, HeuristicExternalEvaluatorConfig, LLMExternalEvaluatorConfig, build_external_evaluator
from src.training.policy_memory import PolicyMemoryConfig, PolicyMemoryStore
from src.training.reward_aggregator import RewardAggregator, RewardAggregatorConfig
from src.training.target_generator import BaseTargetGenerator, GeneratedTarget, LLMTargetGeneratorConfig, build_target_generator
from src.training.teacher_guidance import TeacherGuidanceConfig, TeacherGuidedReranker
from src.utils.settings import build_dataclass_config, get_setting, load_settings

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
    action_bandit_path: str = 'runtime/action_bandit.json'
    use_action_bandit: bool = True
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
        'action_bandit': dict(trace.debug.get('action_bandit', {}) or {}),
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
    action_bandit: ActionBanditStore | None = None,
) -> LearningEpisodeResult:
    settings = load_settings()
    runtime_config = runtime_config or LearningRuntimeConfig()
    external_evaluator = external_evaluator or build_external_evaluator(
        'llm',
        llm_config=build_dataclass_config(LLMExternalEvaluatorConfig, get_setting(settings, 'llm', 'external_evaluation', default={})),
        heuristic_config=build_dataclass_config(HeuristicExternalEvaluatorConfig, get_setting(settings, 'llm', 'heuristic_external', default={})),
    )
    reward_aggregator = reward_aggregator or RewardAggregator(
        build_dataclass_config(RewardAggregatorConfig, get_setting(settings, 'learning', 'reward_aggregator', default={}))
    )
    target_generator = target_generator or build_target_generator(
        'llm',
        config=build_dataclass_config(LLMTargetGeneratorConfig, get_setting(settings, 'llm', 'target_generation', default={})),
    )
    teacher_reranker = teacher_reranker or TeacherGuidedReranker(
        build_dataclass_config(TeacherGuidanceConfig, get_setting(settings, 'teacher_guidance', default={}))
    )

    policy_memory_config = build_dataclass_config(PolicyMemoryConfig, get_setting(settings, 'policy_memory', default={}))
    if runtime_config.use_policy_memory and policy_memory is None:
        policy_memory = PolicyMemoryStore(runtime_config.policy_memory_path, config=policy_memory_config, autoload=True)

    action_bandit_config = build_dataclass_config(ActionBanditConfig, get_setting(settings, 'learning', 'action_bandit', default={}))
    if runtime_config.use_action_bandit and action_bandit is None:
        action_bandit = ActionBanditStore(runtime_config.action_bandit_path, config=action_bandit_config, autoload=True)

    session_id = session_id or new_session_id('learnsess')
    turn_id = new_turn_id('learnturn')
    episode_id = new_episode_id('learnep')

    tokenization_result = build_tokenization_result(
        raw_text=raw_text,
        explicit_words=explicit_words,
        normalizer=normalizer,
    )
    unknown_word_learning = normalizer.maybe_learn_unknown_words(
        raw_text=raw_text,
        tokenization_result=tokenization_result,
    )
    if unknown_word_learning.get('retokenized'):
        tokenization_result = build_tokenization_result(
            raw_text=raw_text,
            explicit_words=explicit_words,
            normalizer=normalizer,
        )
    tokens = list(tokenization_result.normalized_tokens)
    if not tokens:
        raise ValueError('No tokens could be built from the given learning input.')

    input_state = build_input_state(
        raw_text=raw_text,
        tokens=list(tokenization_result.tokens or tokens),
        normalized_tokens=tokens,
        tokenization=list(tokenization_result.tokenization),
        unknown_spans=list(tokenization_result.unknown_spans),
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

    intent_planner_config = build_dataclass_config(IntentPlannerConfig, get_setting(settings, 'pipeline', 'intent_planner', default={}))
    recall_config = build_dataclass_config(SemanticRecallConfig, get_setting(settings, 'pipeline', 'semantic_recall', default={}))
    slot_filler_config = build_dataclass_config(SlotFillerConfig, get_setting(settings, 'pipeline', 'slot_filler', default={}))
    surface_config = build_dataclass_config(SurfaceRealizerConfig, get_setting(settings, 'pipeline', 'surface_realizer', default={}))
    scorer_config = build_dataclass_config(BasicScorerConfig, get_setting(settings, 'pipeline', 'basic_scorer', default={}))

    intent_plan = plan_intent(input_state=input_state, dialogue_state=dialogue_state_current, config=intent_planner_config)
    recall_result = recall_semantics(
        input_state=input_state,
        lexicon=lexicon,
        dialogue_state=dialogue_state_current,
        intent_plan=intent_plan,
        config=recall_config,
    )
    filled_slots = fill_slots(
        input_state=input_state,
        recall_result=recall_result,
        lexicon=lexicon,
        intent_plan=intent_plan,
        dialogue_state=dialogue_state_current,
        config=slot_filler_config,
    )
    surface_plan, base_candidates = realize_surface(
        filled_slots=filled_slots,
        intent_plan=intent_plan,
        lexicon=lexicon,
        config=surface_config,
    )

    policy_memory_matches: List[Dict[str, object]] = []
    memory_candidates: List[RealizationCandidate] = []
    recent_response_texts: List[str] = []
    if runtime_config.use_policy_memory and policy_memory is not None:
        memory_candidates, policy_memory_matches = policy_memory.suggest(
            intent_plan=intent_plan,
            filled_slots=filled_slots,
            existing_texts=[item.text for item in base_candidates],
            limit=max(1, int(runtime_config.policy_memory_limit)),
        )
        recent_response_texts = policy_memory.recent_texts(limit=8, source='selected_response')
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
        config=scorer_config,
        recent_texts=recent_response_texts,
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
        filled_slots=filled_slots,
    )

    response = base_response
    bandit_decision_debug: Dict[str, object] = {}
    bandit_path: Optional[Path] = None
    selected_candidate_index = max(
        range(len(scored_candidates)),
        key=lambda idx: float(scored_candidates[idx].final_score),
    ) if scored_candidates else 0

    if runtime_config.use_action_bandit and action_bandit is not None and scored_candidates:
        bandit_decision = action_bandit.choose_surface_candidate(
            intent_plan=intent_plan,
            filled_slots=filled_slots,
            candidates=scored_candidates,
            teacher_guidance=teacher_guidance,
        )
        bandit_decision_debug = bandit_decision.to_debug_dict()
        selected_candidate_index = max(0, min(len(scored_candidates) - 1, int(bandit_decision.selected_index)))
        selected_candidate = scored_candidates[selected_candidate_index]
        response, _ = choose_best_response(
            input_state=input_state,
            intent_plan=intent_plan,
            filled_slots=filled_slots,
            candidates=[selected_candidate],
            config=scorer_config,
            recent_texts=recent_response_texts,
        )
        LOGGER.info(
            'learning_episode.bandit_selected episode_id=%s selected=%s template_id=%s',
            episode_id,
            response.text,
            selected_candidate.template_id,
        )
    elif teacher_guidance.overridden and 0 <= teacher_guidance.selected_index < len(scored_candidates):
        selected_candidate_index = int(teacher_guidance.selected_index)
        selected_candidate = scored_candidates[selected_candidate_index]
        response, _ = choose_best_response(
            input_state=input_state,
            intent_plan=intent_plan,
            filled_slots=filled_slots,
            candidates=[selected_candidate],
            config=scorer_config,
            recent_texts=recent_response_texts,
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
        'action_bandit': dict(bandit_decision_debug or {}),
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
    if actions and bandit_decision_debug:
        for action in actions:
            if action.stage == 'surface' and action.action_type == 'choose_surface_candidate':
                action.metadata['action_bandit'] = dict(bandit_decision_debug)
                break
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

    if runtime_config.use_action_bandit and action_bandit is not None and response.chosen_candidate is not None:
        action_bandit.update(
            context_key=str(bandit_decision_debug.get('context_key', '')) or action_bandit.build_context_key(intent_plan=intent_plan, filled_slots=filled_slots),
            candidate=response.chosen_candidate,
            reward_total=reward.total,
            reward_internal=reward.internal.total,
            reward_external=reward.external.total,
        )
        bandit_path = action_bandit.save()
        LOGGER.info('learning_episode.action_bandit_saved path=%s episode_id=%s', bandit_path, episode_id)

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
            'tokenization': [dataclass_to_dict(item) for item in tokenization_result.tokenization],
            'unknown_spans': [dataclass_to_dict(item) for item in tokenization_result.unknown_spans],
            'unknown_word_learning': unknown_word_learning,
            'response_score': dataclass_to_dict(response.score),
            'reward': dataclass_to_dict(reward),
            'slot_trace': dataclass_to_dict(slot_trace),
            'evaluation_context': evaluation_context,
            'teacher_target': dataclass_to_dict(generated_target),
            'teacher_guidance': dataclass_to_dict(teacher_guidance),
            'base_response_text': base_response.text,
            'selected_candidate_index': selected_candidate_index,
            'action_bandit': dict(bandit_decision_debug or {}),
            'action_bandit_path': str(bandit_path) if bandit_path else '',
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
