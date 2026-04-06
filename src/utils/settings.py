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
    },
    'pipeline': {
        'intent_planner': {
            'question_min_score': 1.0,
            'question_confidence_base': 0.72,
            'explain_min_score': 1.0,
            'explain_confidence_base': 0.70,
            'confirm_min_score': 1.0,
            'confirm_confidence_base': 0.68,
            'empathy_min_score': 1.0,
            'empathy_confidence_base': 0.66,
            'respond_min_score': 1.0,
            'respond_confidence_base': 0.58,
            'fallback_respond_confidence': 0.51,
            'token_keyword_weight': 1.0,
            'text_keyword_weight': 0.75,
            'question_marker_override_score': 1.5,
            'question_like_ending_bonus': 0.80,
            'question_request_bonus': 0.50,
            'explain_request_bonus': 0.70,
            'empathy_multi_hit_threshold': 2,
            'empathy_multi_hit_bonus': 0.60,
            'confirm_phrase_bonus': 0.60,
            'explain_followup_max_raw_text_length': 20,
            'explain_followup_confidence_floor': 0.57,
            'topic_context_confidence_bonus': 0.08,
            'topic_context_confidence_cap': 0.98,
            'confidence_extra_multiplier': 0.10,
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
        },
        'surface_realizer': {
            'default_style': 'neutral',
            'default_politeness': 'plain',
            'max_candidates': 4,
            'include_question_variant': True,
            'include_soft_variant': True,
        },
        'basic_scorer': {
            'semantic_weight': 0.28,
            'slot_weight': 0.24,
            'grammar_weight': 0.18,
            'retention_weight': 0.16,
            'policy_weight': 0.14,
            'empty_candidate_penalty': 0.40,
            'missing_required_penalty': 0.08,
            'max_grammar_penalty': 0.45,
            'policy_memory_teacher_bonus': 0.22,
            'policy_memory_response_bonus': 0.10,
            'policy_memory_retention_floor': 0.75,
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
        'reward_aggregator': {
            'alpha': 0.8,
            'beta': 0.2,
            'fallback_strategy': 'neutral',
            'neutral_external_score': 0.5,
            'use_power_transform': False,
            'external_power': 1.5,
        },
    },
    'policy_memory': {
        'min_match_score': 0.34,
        'max_suggestions': 4,
        'max_records_per_key': 16,
        'teacher_reward_scale': 1.0,
        'response_reward_scale': 0.6,
        'teacher_source_bonus': 0.08,
        'response_source_bonus': 0.03,
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
