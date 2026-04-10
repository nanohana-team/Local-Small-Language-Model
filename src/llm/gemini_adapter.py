from __future__ import annotations

from .base import BaseLLMAdapter


class GeminiAdapter(BaseLLMAdapter):
    @property
    def provider(self) -> str:
        return "gemini"

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        from google import genai

        client = genai.Client()
        prompt = f"{system_prompt.strip()}\n\n{user_prompt.strip()}".strip()
        config = {}
        if max_output_tokens is not None:
            config["max_output_tokens"] = int(max_output_tokens)
        if temperature is not None:
            config["temperature"] = float(temperature)
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config or None,
        )
        text = getattr(response, "text", None)
        return text or ""
