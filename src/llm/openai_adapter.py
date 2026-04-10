from __future__ import annotations

from .base import BaseLLMAdapter


class OpenAIAdapter(BaseLLMAdapter):
    @property
    def provider(self) -> str:
        return "openai"

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        from openai import OpenAI

        client = OpenAI()
        response = client.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=user_prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        return getattr(response, "output_text", "") or ""
