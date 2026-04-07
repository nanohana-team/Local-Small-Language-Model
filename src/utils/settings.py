from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar

import yaml

T = TypeVar('T')


DEFAULT_SETTINGS: dict[str, Any] = {
    'paths': {
        'lexicon': 'libs/dict.lsdx',
        'trace_dir': 'runtime/traces',
        'dataset_dir': 'runtime/datasets',
        'policy_memory': 'runtime/policy_memory.json',
        'action_bandit': 'runtime/action_bandit.json',
    },
    'llm-api-higher-order': [
        'gemini-3.1-pro',
        'gpt-5.4',
    ],
    'pipeline': {
        'intent_planner': {
            'ask_fact_confidence_base': 0.74,
            'ask_availability_confidence_base': 0.76,
            'ask_recommendation_confidence_base': 0.79,
            'ask_progress_confidence_base': 0.77,
            'smalltalk_expand_confidence_base': 0.73,
            'share_experience_confidence_base': 0.68,
            'explain_confidence_base': 0.72,
            'confirm_confidence_base': 0.70,
            'empathy_confidence_base': 0.68,
            'respond_confidence_base': 0.58,
            'fallback_respond_confidence': 0.52,
            'question_marker_bonus': 1.30,
            'confidence_extra_multiplier': 0.08,
            'topic_context_confidence_bonus': 0.08,
            'topic_context_confidence_cap': 0.98,
            'short_followup_length': 18,
            'progress_recent_bonus': 0.25,
        },
        'semantic_recall': {
            'max_candidates': 24,
            'max_relation_hops': 1,
            'relation_weight_scale': 0.85,
            'axis_weight_scale': 0.75,
            'input_weight': 1.25,
            'prefer_content_words_in_axis': True,
            'axis_probe_limit': 512,
        },
        'slot_filler': {
            'marker_confidence': 0.88,
            'inferred_confidence': 0.58,
            'fallback_confidence': 0.46,
            'use_dialogue_topic_fallback': True,
            'use_recall_topic_fallback': True,
            'recall_topic_min_score': 0.26,
            'allow_axis_only_topic_fallback': False,
        },
        'surface_realizer': {
            'default_style': 'neutral',
            'default_politeness': 'plain',
            'max_candidates': 4,
            'include_question_variant': True,
            'include_soft_variant': True,
        },
        'unknown_word': {
            'enabled': True,
            'min_span_length': 2,
            'max_spans_per_turn': 2,
            'promote_threshold': 3,
            'pending_path': 'runtime/unknown_word_candidates.jsonl',
            'overlay_path': 'runtime/lexicon_overlay.json',
            'temperature': 0.1,
            'max_output_tokens': 220,
            'preferred_models': [],
            'min_overlay_confidence': 0.78,
            'allow_multi_token_surface_promotion': False,
            'reject_function_word_entries': True,
            'reject_unknown_pos_entries': True,
        },
        'basic_scorer': {
            'semantic_weight': 0.26,
            'slot_weight': 0.18,
            'grammar_weight': 0.10,
            'retention_weight': 0.24,
            'policy_weight': 0.22,
            'empty_candidate_penalty': 0.40,
            'missing_required_penalty': 0.08,
            'max_grammar_penalty': 0.45,
            'policy_memory_teacher_bonus': 0.16,
            'policy_memory_response_bonus': 0.08,
            'policy_memory_retention_floor': 0.72,
            'abstract_hedge_penalty': 0.18,
            'low_retention_threshold': 0.34,
            'low_retention_penalty': 0.18,
            'weak_topic_fallback_penalty': 0.10,
        },
    },
    'learning': {
        'episodes': 1,
        'target_mode': 'llm',
        'external_mode': 'llm',
        'input_mode': 'llm',
        'teacher_target_weight': 0.35,
        'report_every': 1,
        'seed_topic': '',
        'seed_text': 'こんにちは。今日はどんな一日になりそうですか。',
        'runtime': {
            'policy_memory_limit': 4,
        },
        'action_bandit': {
            'enabled': True,
            'temperature': 0.85,
            'explore_top_k': 3,
            'scorer_weight': 0.58,
            'teacher_weight': 0.18,
            'learned_weight': 0.18,
            'uncertainty_weight': 0.06,
            'reward_ema_alpha': 0.30,
            'default_value': 0.50,
            'min_probability': 0.02,
            'context_value_slots': ['topic', 'predicate', 'target', 'actor', 'state'],
        },
        'reward_aggregator': {
            'alpha': 0.7,
            'beta': 0.3,
            'fallback_strategy': 'neutral',
            'neutral_external_score': 0.5,
            'use_power_transform': False,
            'external_power': 1.5,
            'mismatch_internal_threshold': 0.76,
            'mismatch_external_threshold': 0.24,
            'mismatch_penalty': 0.24,
        },
    },
    'policy_memory': {
        'min_match_score': 0.55,
        'quarantine_path': 'runtime/policy_memory_quarantine.jsonl',
        'contamination_watch_threshold': 0.45,
        'contamination_danger_threshold': 0.74,
        'max_suggestions': 3,
        'max_records_per_key': 12,
        'teacher_reward_scale': 1.0,
        'response_reward_scale': 0.6,
        'teacher_source_bonus': 0.06,
        'response_source_bonus': 0.02,
        'selected_response_hard_reject_external': 0.24,
        'selected_response_min_reward_total': 0.42,
        'selected_response_low_external_threshold': 0.50,
        'selected_response_low_external_scale': 0.10,
        'min_suggest_weight': 0.12,
        'min_suggest_external': 0.28,
        'min_exact_slot_hits': 1,
        'min_anchor_slot_hits': 1,
        'require_anchor_slot_match': True,
        'teacher_hard_reject_external': 0.05,
        'teacher_min_reward_total': 0.46,
        'min_anchor_mentions_in_text': 1,
        'anchor_free_external_threshold': 0.82,
    },
    'teacher_guidance': {
        'target_weight': 0.35,
        'min_override_delta': 0.015,
        'min_alignment_gain': 0.05,
        'max_blended_regression': 0.01,
    },
    'llm': {
        'target_generation': {
            'temperature': 0.4,
            'max_output_tokens': 160,
        },
        'external_evaluation': {
            'temperature': 0.1,
            'max_output_tokens': 240,
        },
        'input_generation': {
            'temperature': 0.9,
            'max_output_tokens': 120,
            'history_inputs_limit': 8,
            'history_targets_limit': 4,
        },
        'heuristic_external': {
            'non_empty_weight': 0.15,
            'length_weight': 0.15,
            'slot_weight': 0.25,
            'intent_weight': 0.20,
            'target_weight': 0.25,
            'empty_slot_score': 0.5,
            'empty_target_score': 0.5,
            'empty_token_overlap_score': 0.5,
            'question_match_score': 1.0,
            'question_mismatch_score': 0.4,
            'empathy_match_score': 1.0,
            'empathy_mismatch_score': 0.5,
            'default_intent_score': 0.9,
            'punctuated_respond_penalty_score': 0.5,
            'unknown_intent_score': 0.7,
            'exact_echo_penalty': 0.25,
            'partial_echo_penalty': 0.10,
            'empty_length_score': 0.0,
            'short_length_divisor': 4.0,
            'short_length_floor': 4,
            'ideal_length_min': 4,
            'ideal_length_max': 64,
            'long_length_min': 65,
            'long_length_max': 128,
            'long_length_decay_span': 96.0,
            'long_length_base': 64,
            'too_long_score': 0.2,
            'excellent_threshold': 0.85,
            'good_threshold': 0.70,
            'acceptable_threshold': 0.50,
            'weak_threshold': 0.30,
        },
    },
}


def _deep_merge_dict(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_settings(settings_dir: str | Path = 'settings') -> dict[str, Any]:
    settings = deepcopy(DEFAULT_SETTINGS)
    directory = Path(settings_dir)
    if not directory.exists():
        return settings

    for path in sorted(directory.glob('*.yaml')):
        data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        if not isinstance(data, Mapping):
            continue
        settings = _deep_merge_dict(settings, data)
    return settings


def get_setting(settings: Mapping[str, Any], *path: str, default: Any = None) -> Any:
    current: Any = settings
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def build_dataclass_config(cls: type[T], data: Mapping[str, Any] | None = None) -> T:
    payload = dict(data or {})
    allowed = {field.name for field in fields(cls)}
    kwargs = {key: value for key, value in payload.items() if key in allowed}
    return cls(**kwargs)


def apply_arg_defaults(
    args: Any,
    settings: Mapping[str, Any],
    mapping: Sequence[tuple[str, Sequence[str], Any]],
) -> Any:
    for attr_name, path, fallback in mapping:
        if getattr(args, attr_name, None) is None:
            value = get_setting(settings, *path, default=fallback)
            setattr(args, attr_name, value)
    return args
