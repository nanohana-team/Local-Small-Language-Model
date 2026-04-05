from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.llm.llm_router import LLMRouter


class GeminiTeacherPairGenerator:
    """
    1件ずつ、教師あり学習用の input / target ペアを生成する。

    出力形式:
    {
        "input_tokens": [...],
        "target_tokens": [...],
        "target_text": "..."
    }
    """

    def __init__(
        self,
        model_name: str | None = None,
        enabled: bool = True,
        verbose: bool = False,
        max_retry: int = 3,
    ) -> None:
        self.model_name = model_name
        self.enabled = enabled
        self.verbose = verbose
        self.max_retry = max(1, int(max_retry))
        self.router = LLMRouter(verbose=verbose)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    @staticmethod
    def _clean_tokens(values: Any) -> List[str]:
        if not isinstance(values, list):
            return []
        return [str(x).strip() for x in values if str(x).strip()]

    @staticmethod
    def _normalize_text(text: Any) -> str:
        return str(text or "").strip()

    @staticmethod
    def _contains_any_token(tokens: List[str], text: str) -> bool:
        if not tokens or not text:
            return False
        for tok in tokens:
            s = str(tok).strip()
            if not s or s in {"。", "、"}:
                continue
            if s in text:
                return True
        return False

    @staticmethod
    def _ends_like_sentence(text: str) -> bool:
        return text.endswith(("。", "！", "？", "…"))

    def _fallback_pair(self) -> Dict[str, Any]:
        return {
            "input_tokens": ["今日", "疲れた"],
            "target_tokens": ["今日", "は", "お疲れさま", "。"],
            "target_text": "今日はお疲れさま。",
        }

    def _is_valid_pair(self, pair: Dict[str, Any]) -> bool:
        input_tokens = self._clean_tokens(pair.get("input_tokens"))
        target_tokens = self._clean_tokens(pair.get("target_tokens"))
        target_text = self._normalize_text(pair.get("target_text"))

        if not input_tokens or not target_tokens or not target_text:
            return False

        if len(input_tokens) > 8:
            return False
        if len(target_tokens) > 16:
            return False
        if len(target_text) > 48:
            return False

        if not self._ends_like_sentence(target_text):
            return False

        if not self._contains_any_token(target_tokens, target_text):
            return False

        return True

    def _normalize_pair(self, pair: Dict[str, Any]) -> Dict[str, Any]:
        input_tokens = self._clean_tokens(pair.get("input_tokens"))
        target_tokens = self._clean_tokens(pair.get("target_tokens"))
        target_text = self._normalize_text(pair.get("target_text"))

        return {
            "input_tokens": input_tokens,
            "target_tokens": target_tokens,
            "target_text": target_text,
        }

    def generate(self) -> Dict[str, Any]:
        if not self.enabled:
            pair = self._fallback_pair()
            self._log(f"[TEACHER_PAIR] disabled -> fallback {json.dumps(pair, ensure_ascii=False)}")
            return pair

        prompt = {
            "instruction": (
                "Return ONLY valid JSON. No explanation. No markdown. "
                "Generate exactly one Japanese supervised conversation training pair. "
                "input_tokens must be a short tokenized user utterance. "
                "target_tokens and target_text must be a short, natural, helpful reply. "
                "Keep semantic alignment strong. Prefer daily conversation. "
                "Avoid rare words, broken grammar, and long sentences."
            ),
            "task": "generate_teacher_pair",
            "language": "Japanese",
            "constraints": {
                "input_length_min": 1,
                "input_length_max": 6,
                "target_length_min": 2,
                "target_length_max": 12,
                "style": "daily_conversation",
                "tone": "natural_gentle",
                "forbid": [
                    "rare jargon",
                    "broken Japanese",
                    "long explanations",
                    "lists",
                    "meta commentary",
                ],
            },
            "examples": [
                {
                    "input_tokens": ["今日", "疲れた"],
                    "target_tokens": ["今日", "は", "お疲れさま", "。"],
                    "target_text": "今日はお疲れさま。"
                },
                {
                    "input_tokens": ["眠い"],
                    "target_tokens": ["少し", "休もう", "。"],
                    "target_text": "少し休もう。"
                }
            ],
            "output_format": {
                "input_tokens": ["今日", "疲れた"],
                "target_tokens": ["今日", "は", "お疲れさま", "。"],
                "target_text": "今日はお疲れさま。"
            },
        }

        for attempt in range(1, self.max_retry + 1):
            try:
                self._log(f"[TEACHER_PAIR] router_call attempt={attempt} model={self.model_name or 'auto'}")
                result = self.router.generate_json(prompt, model_name=self.model_name)
                pair = self._normalize_pair(result)

                if self._is_valid_pair(pair):
                    self._log(f"[TEACHER_PAIR] accepted {json.dumps(pair, ensure_ascii=False)}")
                    return pair

                self._log(f"[TEACHER_PAIR] invalid pair attempt={attempt}: {json.dumps(pair, ensure_ascii=False)}")
            except Exception as exc:
                self._log(f"[TEACHER_PAIR] failed attempt={attempt}: {type(exc).__name__}:{exc}")

        pair = self._fallback_pair()
        self._log(f"[TEACHER_PAIR] fallback {json.dumps(pair, ensure_ascii=False)}")
        return pair