# src/llm/evaluator_gemini.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import requests


class GeminiEvaluator:
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-lite",
        enabled: bool = True,
        timeout_sec: int = 60,
    ) -> None:
        self.model_name = model_name
        self.enabled = enabled
        self.timeout_sec = timeout_sec
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()

    def _rule_based_fallback(self, episode: Dict[str, Any], error_message: str = "") -> Dict[str, Any]:
        input_tokens = episode.get("input_tokens", [])
        response_text = str(episode.get("response_text", ""))
        initial_core = episode.get("initial_core", [])
        mid_converged = episode.get("mid_converged", [])
        final_expanded = episode.get("final_expanded", [])

        overlap = len(set(input_tokens) & set(final_expanded))
        overlap_score = min(overlap * 10, 30)

        length = len(response_text)
        if 4 <= length <= 30:
            length_score = 20
        elif 1 <= length <= 40:
            length_score = 12
        else:
            length_score = 6

        structure_score = 0
        if len(initial_core) >= 2:
            structure_score += 10
        if len(mid_converged) >= 2:
            structure_score += 10
        if response_text.endswith(("。", "！", "？", "…")):
            structure_score += 10

        total = overlap_score + length_score + structure_score
        total = max(0, min(total, 100))

        return {
            "score_total": float(total),
            "scores": {
                "response_quality": float(length_score + 10),
                "divergence_quality": float(structure_score),
                "convergence_quality": float(overlap_score),
                "efficiency": 50.0,
            },
            "comment": "Rule-based fallback evaluation was used.",
            "model": "rule_based_fallback",
            "used_api": False,
            "error": error_message,
        }

    def _build_prompt(self, episode: Dict[str, Any]) -> str:
        payload = {
            "input_tokens": episode.get("input_tokens", []),
            "initial_core": episode.get("initial_core", []),
            "divergence_steps": episode.get("divergence_steps", []),
            "mid_converged": episode.get("mid_converged", []),
            "final_expanded": episode.get("final_expanded", []),
            "response_text": episode.get("response_text", ""),
            "depth": episode.get("depth", 0),
        }

        prompt = f"""
You are an evaluator for a local small language model.
Evaluate the following thought-search episode.

Return STRICT JSON ONLY with the following schema:
{{
  "score_total": 0-100 number,
  "scores": {{
    "response_quality": 0-100 number,
    "divergence_quality": 0-100 number,
    "convergence_quality": 0-100 number,
    "efficiency": 0-100 number
  }},
  "comment": "short English comment"
}}

Evaluation criteria:
- response_quality: Is the final response natural and minimally coherent?
- divergence_quality: Did the divergence seem meaningfully exploratory rather than random?
- convergence_quality: Did the convergence keep useful tokens and narrow appropriately?
- efficiency: Was the route reasonably efficient for its depth and output?

Episode:
{json.dumps(payload, ensure_ascii=False)}
""".strip()
        return prompt

    def _call_gemini(self, prompt: str) -> Dict[str, Any]:
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:generateContent?key={self.api_key}"
        )
        body = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "topP": 0.8,
                "topK": 20,
                "maxOutputTokens": 512,
                "responseMimeType": "application/json",
            },
        }

        response = requests.post(
            endpoint,
            headers={"Content-Type": "application/json"},
            json=body,
            timeout=self.timeout_sec,
        )
        response.raise_for_status()
        data = response.json()

        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini returned no candidates")

        parts = candidates[0].get("content", {}).get("parts", [])
        text = ""
        for part in parts:
            if "text" in part:
                text += str(part["text"])

        text = text.strip()
        if not text:
            raise RuntimeError("Gemini returned empty text")

        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise RuntimeError("Gemini response is not a JSON object")

        parsed["model"] = self.model_name
        parsed["used_api"] = True
        return parsed

    def evaluate_episode(self, episode: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            return self._rule_based_fallback(episode, error_message="evaluator disabled")

        if not self.api_key:
            return self._rule_based_fallback(episode, error_message="GEMINI_API_KEY not set")

        prompt = self._build_prompt(episode)
        try:
            result = self._call_gemini(prompt)
            result.setdefault("score_total", 50.0)
            result.setdefault("scores", {})
            result.setdefault("comment", "")
            return result
        except Exception as exc:
            return self._rule_based_fallback(episode, error_message=str(exc))