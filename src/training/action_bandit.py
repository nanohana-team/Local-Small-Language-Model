from __future__ import annotations

import json
import logging
import math
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from src.core.schema import FilledSlots, IntentPlan, RealizationCandidate
from src.training.teacher_guidance import TeacherGuidanceResult

LOGGER = logging.getLogger(__name__)
_PUNCT_RE = re.compile(r'[\s、。？！!?,，．・「」『』（）()\[\]{}]+')
_LOW_VALUE_CONTEXT_WORDS = {
    'この', 'その', 'あの', 'それ', 'これ', 'あれ', 'ここ', 'そこ', 'あそこ',
    '何か', '何も', 'もの', 'こと', '一番', '色々', '〇〇', 'どれ', 'どんな',
}
_LOW_VALUE_CONTEXT_ENDINGS = ('は', 'が', 'を', 'に', 'で', 'の', 'も', 'って', 'んだ', 'かな', 'よね', 'っけ')


@dataclass(slots=True)
class ActionBanditConfig:
    enabled: bool = True
    temperature: float = 0.85
    explore_top_k: int = 3
    scorer_weight: float = 0.58
    teacher_weight: float = 0.18
    learned_weight: float = 0.18
    uncertainty_weight: float = 0.06
    reward_ema_alpha: float = 0.30
    default_value: float = 0.50
    min_probability: float = 0.02
    context_value_slots: List[str] = field(default_factory=lambda: ['topic', 'predicate', 'target', 'actor', 'state'])


@dataclass(slots=True)
class BanditArmState:
    arm_key: str
    template_id: str = ''
    text_signature: str = ''
    count: int = 0
    mean_reward: float = 0.0
    ema_reward: float = 0.0
    last_reward_total: float = 0.0
    last_internal: float = 0.0
    last_external: float = 0.0


@dataclass(slots=True)
class BanditCandidateScore:
    index: int
    arm_key: str
    template_id: str
    text: str
    scorer_score: float
    teacher_score: float
    learned_score: float
    uncertainty_bonus: float
    combined_score: float
    probability: float = 0.0
    selected: bool = False


@dataclass(slots=True)
class BanditDecision:
    context_key: str = ''
    selected_index: int = 0
    selected_arm_key: str = ''
    selected_template_id: str = ''
    mode: str = 'disabled'
    probabilities: List[BanditCandidateScore] = field(default_factory=list)

    def to_debug_dict(self) -> Dict[str, Any]:
        return {
            'context_key': self.context_key,
            'selected_index': self.selected_index,
            'selected_arm_key': self.selected_arm_key,
            'selected_template_id': self.selected_template_id,
            'mode': self.mode,
            'candidates': [asdict(item) for item in self.probabilities],
        }


class ActionBanditStore:
    def __init__(
        self,
        path: str | Path,
        *,
        config: ActionBanditConfig | None = None,
        autoload: bool = True,
    ) -> None:
        self.path = Path(path)
        self.config = config or ActionBanditConfig()
        self.contexts: Dict[str, Dict[str, BanditArmState]] = {}
        self.random = random.Random()
        if autoload:
            self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.contexts = {}
            return
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
        except Exception:
            LOGGER.exception('action_bandit.load_failed path=%s', self.path)
            self.contexts = {}
            return

        restored: Dict[str, Dict[str, BanditArmState]] = {}
        for context_key, arm_items in dict(data.get('contexts', {}) or {}).items():
            arm_states: Dict[str, BanditArmState] = {}
            for arm_key, payload in dict(arm_items or {}).items():
                if not isinstance(payload, dict):
                    continue
                arm_states[str(arm_key)] = BanditArmState(
                    arm_key=str(payload.get('arm_key', arm_key) or arm_key),
                    template_id=str(payload.get('template_id', '') or ''),
                    text_signature=str(payload.get('text_signature', '') or ''),
                    count=max(0, int(payload.get('count', 0) or 0)),
                    mean_reward=max(0.0, min(1.0, float(payload.get('mean_reward', 0.0) or 0.0))),
                    ema_reward=max(0.0, min(1.0, float(payload.get('ema_reward', 0.0) or 0.0))),
                    last_reward_total=max(0.0, min(1.0, float(payload.get('last_reward_total', 0.0) or 0.0))),
                    last_internal=max(0.0, min(1.0, float(payload.get('last_internal', 0.0) or 0.0))),
                    last_external=max(0.0, min(1.0, float(payload.get('last_external', 0.0) or 0.0))),
                )
            restored[str(context_key)] = arm_states
        self.contexts = restored

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'contexts': {
                context_key: {arm_key: asdict(state) for arm_key, state in arms.items()}
                for context_key, arms in self.contexts.items()
            }
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        return self.path

    def choose_surface_candidate(
        self,
        *,
        intent_plan: IntentPlan,
        filled_slots: FilledSlots,
        candidates: Sequence[RealizationCandidate],
        teacher_guidance: TeacherGuidanceResult | None = None,
    ) -> BanditDecision:
        if not self.config.enabled or not candidates:
            return BanditDecision(mode='disabled')
        if len(candidates) == 1:
            only = candidates[0]
            return BanditDecision(
                context_key=self.build_context_key(intent_plan=intent_plan, filled_slots=filled_slots),
                selected_index=0,
                selected_arm_key=self.build_arm_key(candidate=only),
                selected_template_id=str(only.template_id or ''),
                mode='single_candidate',
                probabilities=[
                    BanditCandidateScore(
                        index=0,
                        arm_key=self.build_arm_key(candidate=only),
                        template_id=str(only.template_id or ''),
                        text=only.text,
                        scorer_score=float(only.final_score),
                        teacher_score=self._teacher_score(teacher_guidance, 0, float(only.final_score)),
                        learned_score=self.config.default_value,
                        uncertainty_bonus=0.0,
                        combined_score=float(only.final_score),
                        probability=1.0,
                        selected=True,
                    )
                ],
            )

        context_key = self.build_context_key(intent_plan=intent_plan, filled_slots=filled_slots)
        arms = self.contexts.setdefault(context_key, {})
        total_visits = sum(max(0, arm.count) for arm in arms.values())

        scored: List[BanditCandidateScore] = []
        for index, item in enumerate(candidates):
            arm_key = self.build_arm_key(candidate=item)
            arm_state = arms.get(arm_key)
            scorer_score = max(0.0, min(1.0, float(item.final_score)))
            teacher_score = self._teacher_score(teacher_guidance, index, scorer_score)
            learned_score = float(arm_state.ema_reward) if arm_state and arm_state.count > 0 else float(self.config.default_value)
            exploration = self._uncertainty_bonus(total_visits=total_visits, arm_count=arm_state.count if arm_state else 0)
            combined = (
                scorer_score * float(self.config.scorer_weight)
                + teacher_score * float(self.config.teacher_weight)
                + learned_score * float(self.config.learned_weight)
                + exploration * float(self.config.uncertainty_weight)
            )
            scored.append(
                BanditCandidateScore(
                    index=index,
                    arm_key=arm_key,
                    template_id=str(item.template_id or ''),
                    text=item.text,
                    scorer_score=scorer_score,
                    teacher_score=teacher_score,
                    learned_score=learned_score,
                    uncertainty_bonus=exploration,
                    combined_score=combined,
                )
            )

        top_k = max(1, min(len(scored), int(self.config.explore_top_k)))
        ordered = sorted(scored, key=lambda item: (item.combined_score, item.scorer_score), reverse=True)
        selectable = ordered[:top_k]
        probabilities = self._softmax_probabilities(selectable)

        picks = [item.index for item in selectable]
        weights = [probabilities[item.index] for item in selectable]
        chosen_index = self.random.choices(picks, weights=weights, k=1)[0]

        for item in scored:
            item.probability = probabilities.get(item.index, 0.0)
            item.selected = item.index == chosen_index

        chosen_candidate = candidates[chosen_index]
        return BanditDecision(
            context_key=context_key,
            selected_index=chosen_index,
            selected_arm_key=self.build_arm_key(candidate=chosen_candidate),
            selected_template_id=str(chosen_candidate.template_id or ''),
            mode='bandit_softmax',
            probabilities=sorted(scored, key=lambda item: item.index),
        )

    def update(
        self,
        *,
        context_key: str,
        candidate: RealizationCandidate,
        reward_total: float,
        reward_internal: float,
        reward_external: float,
    ) -> None:
        if not self.config.enabled:
            return
        arm_key = self.build_arm_key(candidate=candidate)
        arms = self.contexts.setdefault(str(context_key), {})
        current = arms.get(arm_key)
        if current is None:
            current = BanditArmState(
                arm_key=arm_key,
                template_id=str(candidate.template_id or ''),
                text_signature=self._normalize_text(candidate.text),
            )
            arms[arm_key] = current

        reward_total = max(0.0, min(1.0, float(reward_total)))
        reward_internal = max(0.0, min(1.0, float(reward_internal)))
        reward_external = max(0.0, min(1.0, float(reward_external)))

        current.count += 1
        current.mean_reward = ((current.mean_reward * (current.count - 1)) + reward_total) / float(current.count)
        alpha = max(0.0, min(1.0, float(self.config.reward_ema_alpha)))
        if current.count == 1:
            current.ema_reward = reward_total
        else:
            current.ema_reward = ((1.0 - alpha) * current.ema_reward) + (alpha * reward_total)
        current.last_reward_total = reward_total
        current.last_internal = reward_internal
        current.last_external = reward_external
        current.template_id = str(candidate.template_id or current.template_id)
        if not current.text_signature:
            current.text_signature = self._normalize_text(candidate.text)

    def build_context_key(self, *, intent_plan: IntentPlan, filled_slots: FilledSlots) -> str:
        parts: List[str] = [f'intent={intent_plan.intent}', f'policy={intent_plan.response_policy_hint}']
        predicate_value = str(filled_slots.frame.predicate or '').strip()
        normalized_predicate = self._normalize_text(predicate_value) if predicate_value else ''
        if predicate_value and not self._is_low_value_context_value(predicate_value):
            parts.append(f'predicate={predicate_value}')
        for slot_name in self.config.context_value_slots:
            slot = filled_slots.values.get(slot_name)
            slot_value = str(slot.value).strip() if slot else ''
            normalized_slot_value = self._normalize_text(slot_value) if slot_value else ''
            if slot_name == 'predicate' and normalized_slot_value == normalized_predicate:
                continue
            if slot_value and not self._is_low_value_context_value(slot_value):
                parts.append(f'{slot_name}={normalized_slot_value}')
        if filled_slots.missing_required:
            parts.append('missing=' + ','.join(sorted(str(name) for name in filled_slots.missing_required)))
        return '|'.join(parts)

    def build_arm_key(self, *, candidate: RealizationCandidate) -> str:
        template = str(candidate.template_id or '').strip()
        if template:
            return f'template:{template}'
        return f'text:{self._normalize_text(candidate.text)}'

    def _teacher_score(
        self,
        teacher_guidance: TeacherGuidanceResult | None,
        index: int,
        fallback: float,
    ) -> float:
        if teacher_guidance is None:
            return fallback
        for item in teacher_guidance.rankings:
            if item.index == index:
                return max(0.0, min(1.0, float(item.blended_score)))
        return fallback

    def _uncertainty_bonus(self, *, total_visits: int, arm_count: int) -> float:
        return math.sqrt(math.log(total_visits + 2.0) / float(arm_count + 1))

    def _softmax_probabilities(self, candidates: Sequence[BanditCandidateScore]) -> Dict[int, float]:
        if not candidates:
            return {}
        temperature = max(0.05, float(self.config.temperature))
        raw_values = [item.combined_score / temperature for item in candidates]
        max_value = max(raw_values)
        exp_values = [math.exp(value - max_value) for value in raw_values]
        total = sum(exp_values) or 1.0
        probs = [value / total for value in exp_values]

        floor = max(0.0, min(0.25, float(self.config.min_probability)))
        if floor > 0.0:
            adjusted = [max(floor, prob) for prob in probs]
            norm = sum(adjusted) or 1.0
            probs = [prob / norm for prob in adjusted]

        probability_map = {item.index: prob for item, prob in zip(candidates, probs)}
        return probability_map

    def _normalize_text(self, text: str) -> str:
        return _PUNCT_RE.sub('', str(text or '')).strip().lower()

    def _is_low_value_context_value(self, value: str) -> bool:
        text = str(value or '').strip()
        if not text:
            return True
        if text in _LOW_VALUE_CONTEXT_WORDS:
            return True
        if text.startswith('〇〇'):
            return True
        if len(text) <= 2 and all('ぁ' <= ch <= 'ん' or ch == 'ー' for ch in text):
            return True
        if len(text) <= 3 and any(text.endswith(ending) for ending in _LOW_VALUE_CONTEXT_ENDINGS):
            return True
        return False
