from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from src.training.llm_gateway import LLMGateway

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class GeneratedTarget:
    text: str
    source: str
    model: str = ''
    metadata: dict[str, Any] | None = None


class BaseTargetGenerator(Protocol):
    def generate(
        self,
        *,
        user_input: str,
        intent: str,
        slots: Mapping[str, str] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> GeneratedTarget:
        ...


@dataclass(slots=True)
class EchoTargetGenerator:
    source: str = 'echo'

    def generate(
        self,
        *,
        user_input: str,
        intent: str,
        slots: Mapping[str, str] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> GeneratedTarget:
        return GeneratedTarget(
            text=str(user_input or '').strip(),
            source=self.source,
            metadata={'intent': intent, 'slots': dict(slots or {})},
        )


@dataclass(slots=True)
class LLMTargetGeneratorConfig:
    temperature: float = 0.4
    max_output_tokens: int = 160


class LLMTargetGenerator:
    def __init__(
        self,
        gateway: LLMGateway | None = None,
        config: LLMTargetGeneratorConfig | None = None,
    ) -> None:
        self.gateway = gateway or LLMGateway()
        self.config = config or LLMTargetGeneratorConfig()

    def generate(
        self,
        *,
        user_input: str,
        intent: str,
        slots: Mapping[str, str] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> GeneratedTarget:
        slot_lines = '\n'.join(f'- {k}: {v}' for k, v in dict(slots or {}).items()) or '- (none)'
        system_prompt = (
            'あなたは日本語対話学習用の外部教師です。'
            'ユーザー入力に対して、学習の正解例として使える自然で短い応答を1つだけ作成してください。'
            '余計な説明、見出し、箇条書き、引用符は不要です。'
            '返答本文だけを出力してください。'
        )
        user_prompt = (
            f'意図: {intent}\n'
            f'抽出スロット:\n{slot_lines}\n\n'
            f'ユーザー入力:\n{user_input}\n\n'
            '条件:\n'
            '- 日本語で返す\n'
            '- 1〜2文、簡潔\n'
            '- 入力意図に自然に答える\n'
            '- ただのオウム返しは避ける\n'
            '- 学習用の理想応答を1つだけ返す\n'
        )
        response = self.gateway.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            purpose='target_generation',
            temperature=float(self.config.temperature),
            max_output_tokens=max(1, int(self.config.max_output_tokens)),
        )
        text = self._clean_text(response.text)
        LOGGER.info('target_generator.llm.done model=%s text=%s', response.model, text)
        return GeneratedTarget(
            text=text,
            source=response.provider,
            model=response.model,
            metadata={'raw': response.raw, 'intent': intent, 'slots': dict(slots or {})},
        )

    def _clean_text(self, text: str) -> str:
        cleaned = str(text or '').strip()
        cleaned = cleaned.strip('"\'“”')
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if not lines:
            return ''
        return ' '.join(lines)


def build_target_generator(
    mode: str = 'llm',
    *,
    gateway: LLMGateway | None = None,
    config: LLMTargetGeneratorConfig | None = None,
) -> BaseTargetGenerator:
    normalized = str(mode or 'llm').strip().lower()
    if normalized in {'llm', 'teacher', 'external-teacher'}:
        return LLMTargetGenerator(gateway=gateway, config=config)
    if normalized in {'echo', 'none', 'disabled', 'off'}:
        return EchoTargetGenerator()
    raise ValueError(f'Unsupported target generator mode: {mode}')
