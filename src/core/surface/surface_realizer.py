from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.core.schema import (
    FilledSlots,
    IntentPlan,
    LexiconContainer,
    LexiconEntry,
    RealizationCandidate,
    SurfacePlan,
)

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SurfaceRealizerConfig:
    default_style: str = "neutral"
    default_politeness: str = "plain"
    max_candidates: int = 5
    include_question_variant: bool = True
    include_soft_variant: bool = True


class SurfaceRealizer:
    def __init__(self, config: Optional[SurfaceRealizerConfig] = None) -> None:
        self.config = config or SurfaceRealizerConfig()

    def realize(
        self,
        filled_slots: FilledSlots,
        intent_plan: Optional[IntentPlan] = None,
        lexicon: Optional[LexiconContainer] = None,
    ) -> Tuple[SurfacePlan, List[RealizationCandidate]]:
        intent_plan = intent_plan or IntentPlan(intent="unknown")
        LOGGER.debug(
            'surface_realizer.start intent=%s predicate=%s values=%s missing_required=%s',
            intent_plan.intent,
            filled_slots.frame.predicate,
            {k: v.value for k, v in filled_slots.values.items()},
            filled_slots.missing_required,
        )

        plan = self._build_surface_plan(filled_slots=filled_slots, intent_plan=intent_plan)
        texts = self._build_texts(filled_slots=filled_slots, intent_plan=intent_plan, lexicon=lexicon)
        unique_texts = self._dedupe_keep_order(texts)
        slot_coverage = self._slot_coverage(filled_slots)
        candidates: List[RealizationCandidate] = []
        for index, text in enumerate(unique_texts[: self.config.max_candidates], start=1):
            if not text:
                continue
            candidates.append(
                RealizationCandidate(
                    text=text,
                    token_sequence=self._simple_tokenize(text),
                    template_id=f'{plan.template_id}_v{index}',
                    grammar_violations=self._quick_grammar_checks(text),
                    slot_coverage=slot_coverage,
                    semantic_score=self._estimate_semantic_score(text=text, filled_slots=filled_slots, intent_plan=intent_plan),
                    final_score=0.0,
                )
            )

        if not candidates:
            fallback_text = self._fallback_text(intent_plan=intent_plan, filled_slots=filled_slots)
            candidates = [
                RealizationCandidate(
                    text=fallback_text,
                    token_sequence=self._simple_tokenize(fallback_text),
                    template_id='fallback_minimal',
                    grammar_violations=[],
                    slot_coverage=slot_coverage,
                    semantic_score=0.40,
                    final_score=0.0,
                )
            ]

        LOGGER.debug('surface_realizer.result template_id=%s texts=%s', plan.template_id, [candidate.text for candidate in candidates])
        return plan, candidates

    def _build_surface_plan(self, filled_slots: FilledSlots, intent_plan: IntentPlan) -> SurfacePlan:
        style_map = {
            'empathy': 'warm',
            'explain': 'informative',
            'confirm': 'checking',
            'ask_fact': 'grounded',
            'ask_availability': 'planning',
            'ask_recommendation': 'guiding',
            'ask_progress': 'tracking',
            'smalltalk_expand': 'friendly',
            'share_experience': 'friendly',
        }
        return SurfacePlan(
            template_id=f"{intent_plan.intent}_{filled_slots.frame.predicate_type or 'generic'}",
            style=style_map.get(intent_plan.intent, self.config.default_style),
            politeness=self.config.default_politeness,
            sentence_count=1,
            order=self._default_order_for_intent(intent_plan.intent),
            auxiliaries=[],
            note=f'surface_plan_for:{intent_plan.intent}',
        )

    def _build_texts(self, filled_slots: FilledSlots, intent_plan: IntentPlan, lexicon: Optional[LexiconContainer]) -> List[str]:
        if filled_slots.missing_required:
            clarify = self._realize_clarify(filled_slots=filled_slots, intent_plan=intent_plan, lexicon=lexicon)
            if clarify:
                return clarify

        if intent_plan.intent == 'empathy':
            return self._realize_empathy(filled_slots, lexicon)
        if intent_plan.intent == 'confirm':
            return self._realize_confirm(filled_slots, lexicon)
        if intent_plan.intent == 'explain':
            return self._realize_explain(filled_slots, lexicon)
        if intent_plan.intent == 'ask_recommendation':
            return self._realize_ask_recommendation(filled_slots, lexicon)
        if intent_plan.intent == 'ask_progress':
            return self._realize_ask_progress(filled_slots, lexicon)
        if intent_plan.intent == 'ask_availability':
            return self._realize_ask_availability(filled_slots, lexicon)
        if intent_plan.intent == 'smalltalk_expand':
            return self._realize_smalltalk_expand(filled_slots, lexicon)
        if intent_plan.intent == 'share_experience':
            return self._realize_share_experience(filled_slots, lexicon)
        if intent_plan.intent == 'ask_fact':
            return self._realize_ask_fact(filled_slots, lexicon)
        return self._realize_respond(filled_slots, lexicon)

    def _realize_clarify(self, filled_slots: FilledSlots, intent_plan: IntentPlan, lexicon: Optional[LexiconContainer]) -> List[str]:
        topic = self._slot_surface(filled_slots, 'topic', lexicon)
        if intent_plan.intent == 'ask_recommendation':
            return [
                'どんなジャンルのおすすめが知りたいですか？',
                f'{topic}について、もう少し条件を教えてください。' if topic else '',
            ]
        if intent_plan.intent == 'ask_progress':
            return [
                'どの件の進み具合か教えてください。',
                f'{topic}についてなら、今どの段階か教えてもらえると答えやすいです。' if topic else '',
            ]
        if intent_plan.intent == 'ask_availability':
            return [
                'いつ頃の予定を見たいですか？',
                '候補の日付や時間があれば、それに合わせて整理できます。',
            ]
        return ['もう少し具体的に教えてもらえると答えやすいです。']

    def _realize_respond(self, filled_slots: FilledSlots, lexicon: Optional[LexiconContainer]) -> List[str]:
        topic = self._slot_surface(filled_slots, 'topic', lexicon)
        state = self._slot_surface(filled_slots, 'state', lexicon)
        predicate_polite = self._slot_surface(filled_slots, 'predicate', lexicon, preferred_form='polite')
        texts: List[str] = []
        if topic and state:
            texts.append(f'{topic}は{state}ですね。')
        if topic and predicate_polite:
            texts.append(f'{topic}については{predicate_polite}。')
        if topic:
            texts.append(f'{topic}ですね。')
        if state:
            texts.append(f'{state}です。')
        texts.append('そうですね。')
        return texts

    def _realize_ask_fact(self, filled_slots: FilledSlots, lexicon: Optional[LexiconContainer]) -> List[str]:
        topic = self._slot_surface(filled_slots, 'topic', lexicon)
        time = self._slot_surface(filled_slots, 'time', lexicon)
        state = self._slot_surface(filled_slots, 'state', lexicon)
        predicate = self._meaningful_predicate(filled_slots, lexicon, preferred_form='plain')
        texts: List[str] = []
        if topic and time and state:
            texts.append(f'{time}の{topic}は{state}という見方が近そうです。')
        if topic and state:
            texts.append(f'{topic}は{state}という理解が近いです。')
        if topic and predicate:
            texts.append(f'{topic}については、{predicate}方向で見るのが自然です。')
        if topic:
            texts.append(f'{topic}についてなら、その話として受け取れます。')
        texts.append('今の情報だけだと断定まではしにくいですが、話題としてはそこです。')
        return texts

    def _realize_ask_availability(self, filled_slots: FilledSlots, lexicon: Optional[LexiconContainer]) -> List[str]:
        topic = self._slot_surface(filled_slots, 'topic', lexicon)
        if topic in {'この', 'その', 'あの'}:
            topic = ''
        time = self._slot_surface(filled_slots, 'time', lexicon)
        location = self._slot_surface(filled_slots, 'location', lexicon)
        texts: List[str] = []
        if time and topic:
            texts.append(f'{time}の{topic}について予定を確認したい感じですね。')
        if time:
            texts.append(f'{time}の都合を見たいということですね。')
        if topic and location:
            texts.append(f'{location}での{topic}について予定を合わせる流れですね。')
        if topic:
            texts.append(f'{topic}について都合を合わせたい感じですね。')
        texts.append('候補の日時があると、もっと具体的に整理できます。')
        return texts

    def _realize_ask_recommendation(self, filled_slots: FilledSlots, lexicon: Optional[LexiconContainer]) -> List[str]:
        topic = self._slot_surface(filled_slots, 'topic', lexicon)
        target = self._slot_surface(filled_slots, 'target', lexicon)
        state = self._slot_surface(filled_slots, 'state', lexicon)
        focus = topic or target
        texts: List[str] = []
        if not focus:
            return ['どんなジャンルのおすすめが知りたいですか？', '条件が一つあるとおすすめしやすいです。']
        if focus and state:
            texts.append(f'{focus}なら、{state}寄りのものから探すと合いやすそうです。')
        if focus:
            texts.append(f'{focus}なら、まず定番で評判の良いものから見るのがよさそうです。')
            texts.append(f'{focus}なら、好みが分かれやすいので条件を一つ決めると選びやすいです。')
        texts.append('おすすめなら、ジャンルや条件が一つあるとかなり絞れます。')
        return texts

    def _realize_ask_progress(self, filled_slots: FilledSlots, lexicon: Optional[LexiconContainer]) -> List[str]:
        topic = self._slot_surface(filled_slots, 'topic', lexicon)
        state = self._slot_surface(filled_slots, 'state', lexicon)
        time = self._slot_surface(filled_slots, 'time', lexicon)
        texts: List[str] = []
        if topic and state:
            texts.append(f'{topic}は今のところ{state}という段階に見えます。')
        if topic and time:
            texts.append(f'{time}時点の{topic}の状況を見たいということですね。')
        if topic:
            texts.append(f'{topic}の進み具合を確認したい感じですね。')
        texts.append('今どの段階かが分かると、もう少し踏み込んで答えられます。')
        return texts

    def _realize_smalltalk_expand(self, filled_slots: FilledSlots, lexicon: Optional[LexiconContainer]) -> List[str]:
        topic = self._slot_surface(filled_slots, 'topic', lexicon)
        target = self._slot_surface(filled_slots, 'target', lexicon)
        focus = topic or target
        texts: List[str] = []
        if not focus:
            return ['どんなジャンルのおすすめが知りたいですか？', '条件が一つあるとおすすめしやすいです。']
        if focus:
            texts.append(f'いいですね。{focus}の話、もう少し広げたいです。')
            texts.append(f'{focus}なら楽しそうですね。')
        texts.append('いいですね。その流れ、けっこう好きです。')
        texts.append('楽しそうな話ですね。')
        return texts

    def _realize_share_experience(self, filled_slots: FilledSlots, lexicon: Optional[LexiconContainer]) -> List[str]:
        topic = self._slot_surface(filled_slots, 'topic', lexicon)
        state = self._slot_surface(filled_slots, 'state', lexicon)
        texts: List[str] = []
        if topic and state:
            texts.append(f'{topic}が{state}だったんですね。ちょっと気になります。')
        if topic:
            texts.append(f'{topic}の話なんですね。続きが気になります。')
        texts.append('それ、ちょっと詳しく聞きたいです。')
        texts.append('へえ、それは気になる流れですね。')
        return texts

    def _realize_explain(self, filled_slots: FilledSlots, lexicon: Optional[LexiconContainer]) -> List[str]:
        topic = self._slot_surface(filled_slots, 'topic', lexicon)
        predicate = self._slot_surface(filled_slots, 'predicate', lexicon, preferred_form='plain')
        state = self._slot_surface(filled_slots, 'state', lexicon)
        cause = self._slot_surface(filled_slots, 'cause', lexicon)
        texts: List[str] = []
        if topic and state:
            texts.append(f'{topic}は{state}という状態として捉えられます。')
        if topic and predicate:
            texts.append(f'{topic}は、要するに{predicate}方向の話です。')
        if topic and cause:
            texts.append(f'{topic}は{cause}が背景にあると考えられます。')
        if topic:
            texts.append(f'{topic}について整理すると、そういう意味合いです。')
        texts.append('ポイントを分けると理解しやすいです。')
        return texts

    def _realize_confirm(self, filled_slots: FilledSlots, lexicon: Optional[LexiconContainer]) -> List[str]:
        topic = self._slot_surface(filled_slots, 'topic', lexicon)
        state = self._slot_surface(filled_slots, 'state', lexicon)
        texts: List[str] = []
        if topic and state:
            texts.append(f'{topic}は{state}という認識で大丈夫です。')
        if topic:
            texts.append(f'{topic}については、その理解で問題ありません。')
        texts.append('その理解で大丈夫です。')
        return texts

    def _realize_empathy(self, filled_slots: FilledSlots, lexicon: Optional[LexiconContainer]) -> List[str]:
        state = self._slot_surface(filled_slots, 'state', lexicon)
        cause = self._slot_surface(filled_slots, 'cause', lexicon)
        topic = self._slot_surface(filled_slots, 'topic', lexicon)
        texts: List[str] = []
        if state and cause:
            texts.append(f'{cause}があって{state}なんですね。')
        if topic and state:
            texts.append(f'{topic}のことで{state}なんですね。')
        if state:
            texts.append(f'{state}のはしんどいですよね。')
        texts.append('それは大変でしたね。')
        return texts

    def _meaningful_predicate(self, filled_slots: FilledSlots, lexicon: Optional[LexiconContainer], preferred_form: str = 'plain') -> str:
        predicate = self._slot_surface(filled_slots, 'predicate', lexicon, preferred_form=preferred_form)
        if predicate in {'ある', 'する', '行', '行く', 'なる', 'できる', 'どう'}:
            return ''
        return predicate

    def _slot_surface(self, filled_slots: FilledSlots, slot_name: str, lexicon: Optional[LexiconContainer], preferred_form: str = 'plain') -> str:
        slot = filled_slots.values.get(slot_name)
        if slot is None:
            return ''
        word = str(slot.value).strip()
        if not word:
            return ''
        if lexicon is None:
            return word
        entry = lexicon.entries.get(word)
        if entry is None:
            return word
        return self._realize_entry(entry, preferred_form=preferred_form, slot_name=slot_name)

    def _realize_entry(self, entry: LexiconEntry, preferred_form: str, slot_name: str) -> str:
        if entry.grammar.pos in {'verb_stem', 'verb'} or entry.entry_type == 'stem':
            form = preferred_form
            if preferred_form == 'plain' and slot_name in {'state', 'topic'}:
                form = 'plain'
            return entry.get_surface(form)
        if entry.grammar.pos == 'adjective_i':
            return entry.get_surface('plain')
        if entry.grammar.pos == 'adjective_na':
            return entry.word
        if entry.grammar.pos in {'copula', 'auxiliary'}:
            return entry.get_surface(preferred_form)
        return entry.get_surface(preferred_form)

    def _default_order_for_intent(self, intent: str) -> List[str]:
        if intent == 'empathy':
            return ['cause', 'topic', 'state', 'predicate']
        if intent in {'ask_recommendation', 'ask_progress', 'ask_availability', 'ask_fact'}:
            return ['topic', 'target', 'state', 'time', 'location', 'predicate']
        if intent == 'explain':
            return ['topic', 'predicate', 'cause', 'location', 'time', 'state']
        if intent == 'confirm':
            return ['topic', 'state', 'predicate']
        return ['topic', 'actor', 'target', 'predicate', 'state']

    def _fallback_text(self, intent_plan: IntentPlan, filled_slots: FilledSlots) -> str:
        topic = self._get_value(filled_slots, 'topic')
        if intent_plan.intent == 'empathy':
            return 'つらそうですね。'
        if intent_plan.intent == 'confirm':
            return 'その理解で大丈夫です。'
        if intent_plan.intent == 'ask_recommendation':
            return '条件が分かるとおすすめしやすいです。'
        if intent_plan.intent == 'ask_progress':
            return 'どの件か分かると状況を整理しやすいです。'
        if topic:
            return f'{topic}についての話ですね。'
        return 'そうですね。'

    def _get_value(self, filled_slots: FilledSlots, slot_name: str) -> str:
        slot = filled_slots.values.get(slot_name)
        return str(slot.value).strip() if slot else ''

    def _slot_coverage(self, filled_slots: FilledSlots) -> float:
        if not filled_slots.frame.constraints:
            return 1.0 if filled_slots.values else 0.0
        total = len(filled_slots.frame.constraints)
        filled = sum(1 for constraint in filled_slots.frame.constraints if constraint.name in filled_slots.values and filled_slots.values[constraint.name].value)
        return filled / float(total) if total > 0 else 0.0

    def _quick_grammar_checks(self, text: str) -> List[str]:
        violations: List[str] = []
        if '。。' in text:
            violations.append('double_period')
        if 'です。です' in text:
            violations.append('duplicated_copula')
        if '、、' in text:
            violations.append('duplicated_punctuation')
        return violations

    def _estimate_semantic_score(self, text: str, filled_slots: FilledSlots, intent_plan: IntentPlan) -> float:
        score = 0.42
        for slot in filled_slots.values.values():
            if slot.value and slot.value in text:
                score += 0.11
        if intent_plan.intent in {'ask_recommendation', 'ask_progress', 'ask_availability'} and '？' in text:
            score += 0.06
        if intent_plan.intent == 'smalltalk_expand' and ('いいですね' in text or '楽しそう' in text):
            score += 0.10
        if intent_plan.intent == 'share_experience' and ('聞きたい' in text or '気になる' in text):
            score += 0.10
        if 'ことが考えられます' in text or '確認できる範囲ではそのように見えます' in text:
            score -= 0.28
        return max(0.0, min(1.0, score))

    def _simple_tokenize(self, text: str) -> List[str]:
        return [chunk for chunk in text.replace('。', ' 。').replace('、', ' 、').replace('？', ' ？').split() if chunk]

    def _dedupe_keep_order(self, texts: List[str]) -> List[str]:
        result: List[str] = []
        seen = set()
        for text in texts:
            normalized = self._normalize_text(text)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
        return result

    def _normalize_text(self, text: str) -> str:
        text = str(text).strip()
        if not text:
            return ''
        if not text.endswith(('。', '？', '!', '！')):
            text += '。'
        return text


def realize_surface(
    filled_slots: FilledSlots,
    intent_plan: Optional[IntentPlan] = None,
    lexicon: Optional[LexiconContainer] = None,
    config: Optional[SurfaceRealizerConfig] = None,
) -> Tuple[SurfacePlan, List[RealizationCandidate]]:
    realizer = SurfaceRealizer(config=config)
    return realizer.realize(filled_slots=filled_slots, intent_plan=intent_plan, lexicon=lexicon)
