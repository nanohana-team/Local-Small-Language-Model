# src/llm/evaluator_gemini.py
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from src.llm.llm_router import LLMRouter


class GeminiEvaluator:
    def __init__(
        self,
        model_name: str | None = None,
        enabled: bool = True,
        verbose: bool = True,
    ):
        self.enabled = enabled
        self.model_name = model_name
        self.verbose = verbose
        self.router = LLMRouter(verbose=verbose)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def _clip01(self, value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    def _to_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default

    # =========================================================
    # learn mode
    # =========================================================

    def evaluate_episode(self, episode: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            return self._fallback_episode_evaluation(episode=episode, reason="disabled")

        prompt = {
            "instruction": (
                "Return ONLY valid JSON object. No explanation. No markdown. "
                "Evaluate this episode for reinforcement learning."
            ),
            "task": "evaluate_episode",
            "layer": "core_thought",
            "input_tokens": [str(x) for x in episode.get("input_tokens", [])],
            "initial_core": [str(x) for x in episode.get("initial_core", [])],
            "mid_converged": [str(x) for x in episode.get("mid_converged", [])],
            "final_expanded": [str(x) for x in episode.get("final_expanded", [])],
            "response_text": str(episode.get("response_text", "")),
            "criteria": [
                "response_quality",
                "divergence_quality",
                "convergence_quality",
                "thought_efficiency",
                "naturalness",
            ],
            "output_format": {
                "score_response": 0.0,
                "score_divergence": 0.0,
                "score_convergence": 0.0,
                "score_efficiency": 0.0,
                "score_naturalness": 0.0,
                "score_total": 0.0,
                "reason": "",
            },
        }

        try:
            self._log(f"[EVAL][episode] router_call model={self.model_name or 'auto'}")
            result = self.router.generate_json(prompt, model_name=self.model_name)
            return self._normalize_episode_evaluation(result)
        except Exception as exc:
            self._log(f"[EVAL][episode] fallback reason={type(exc).__name__}:{exc}")
            return self._fallback_episode_evaluation(
                episode=episode,
                reason=f"fallback:{type(exc).__name__}:{exc}",
            )

    def _normalize_episode_evaluation(self, result: Dict[str, Any]) -> Dict[str, Any]:
        score_response = self._clip01(self._to_float(result.get("score_response"), 0.5))
        score_divergence = self._clip01(self._to_float(result.get("score_divergence"), 0.5))
        score_convergence = self._clip01(self._to_float(result.get("score_convergence"), 0.5))
        score_efficiency = self._clip01(self._to_float(result.get("score_efficiency"), 0.5))
        score_naturalness = self._clip01(self._to_float(result.get("score_naturalness"), 0.5))

        raw_total = result.get("score_total")
        if raw_total is None:
            score_total = (
                score_response * 0.30
                + score_divergence * 0.20
                + score_convergence * 0.20
                + score_efficiency * 0.15
                + score_naturalness * 0.15
            )
        else:
            score_total = self._clip01(self._to_float(raw_total, 0.5))

        return {
            "score_response": round(score_response, 6),
            "score_divergence": round(score_divergence, 6),
            "score_convergence": round(score_convergence, 6),
            "score_efficiency": round(score_efficiency, 6),
            "score_naturalness": round(score_naturalness, 6),
            "score_total": round(score_total, 6),
            "reason": str(result.get("reason", "")),
        }
    
    def evaluate_recursive_state(
        self,
        original_input_tokens: List[str],
        current_input_tokens: List[str],
        raw_output_tokens: List[str],
        normalized_candidate_tokens: List[str],
        step_index: int,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "accepted": False,
                "score_total": 0.0,
                "reason": "disabled",
                "step": step_index,
            }

        prompt = {
            "instruction": (
                "Return ONLY valid JSON object. No explanation. No markdown. "
                "Evaluate whether this recursive thought step is good enough to stop."
            ),
            "task": "evaluate_recursive_state",
            "layer": "core_thought_recursive",
            "original_input_tokens": [str(x) for x in original_input_tokens],
            "current_input_tokens": [str(x) for x in current_input_tokens],
            "raw_output_tokens": [str(x) for x in raw_output_tokens],
            "normalized_candidate_tokens": [str(x) for x in normalized_candidate_tokens],
            "step_index": int(step_index),
            "criteria": [
                "meaning_preservation",
                "thought_progress",
                "stability",
                "stop_readiness",
            ],
            "output_format": {
                "accepted": False,
                "score_total": 0.0,
                "reason": "",
            },
        }

        try:
            self._log(f"[EVAL][recursive_state] router_call model={self.model_name or 'auto'}")
            result = self.router.generate_json(prompt, model_name=self.model_name)

            accepted = bool(result.get("accepted", False))
            score_total = self._clip01(self._to_float(result.get("score_total"), 0.0))

            return {
                "accepted": accepted,
                "score_total": round(score_total, 6),
                "reason": str(result.get("reason", "")),
                "step": step_index,
            }
        except Exception as exc:
            self._log(f"[EVAL][recursive_state] fallback reason={type(exc).__name__}:{exc}")
            return self._fallback_recursive_state(
                original_input_tokens=original_input_tokens,
                current_input_tokens=current_input_tokens,
                raw_output_tokens=raw_output_tokens,
                normalized_candidate_tokens=normalized_candidate_tokens,
                step_index=step_index,
                reason=f"fallback:{type(exc).__name__}:{exc}",
            )
        
    # =========================================================
    # verbal mode
    # =========================================================

    def evaluate_verbal_candidates(
        self,
        input_tokens: List[str],
        final_tokens: List[str],
        candidates: List[str],
    ) -> Dict[str, Any]:
        if not candidates:
            return {"best_text": "", "scores": [], "reason": "no_candidates"}

        if not self.enabled:
            return self._fallback_verbal_candidates(
                input_tokens=input_tokens,
                final_tokens=final_tokens,
                candidates=candidates,
                reason="disabled",
            )

        prompt = {
            "instruction": (
                "Return ONLY valid JSON object. No explanation. No markdown. "
                "Choose the best Japanese sentence candidate."
            ),
            "task": "evaluate_text_candidates",
            "layer": "verbal_surface",
            "input_tokens": [str(x) for x in input_tokens],
            "thought_tokens": [str(x) for x in final_tokens],
            "candidates": [str(x) for x in candidates],
            "criteria": [
                "naturalness",
                "meaning_preservation",
                "fluency",
                "conciseness",
            ],
            "output_format": {
                "best_index": 0,
                "scores": [0.0],
                "reason": "",
            },
        }

        try:
            self._log(f"[EVAL][verbal_candidates] router_call model={self.model_name or 'auto'}")
            result = self.router.generate_json(prompt, model_name=self.model_name)
            return self._normalize_verbal_candidate_evaluation(result, candidates)
        except Exception as exc:
            self._log(f"[EVAL][verbal_candidates] fallback reason={type(exc).__name__}:{exc}")
            return self._fallback_verbal_candidates(
                input_tokens=input_tokens,
                final_tokens=final_tokens,
                candidates=candidates,
                reason=f"fallback:{type(exc).__name__}:{exc}",
            )

    def _normalize_verbal_candidate_evaluation(
        self,
        result: Dict[str, Any],
        candidates: Sequence[str],
    ) -> Dict[str, Any]:
        candidate_list = [str(x) for x in candidates]
        if not candidate_list:
            return {"best_text": "", "scores": [], "reason": "no_candidates"}

        raw_scores = result.get("scores", [])
        if not isinstance(raw_scores, list):
            raw_scores = []

        scores: List[float] = []
        for value in raw_scores[: len(candidate_list)]:
            scores.append(round(self._clip01(self._to_float(value, 0.0)), 6))
        while len(scores) < len(candidate_list):
            scores.append(0.0)

        try:
            idx = int(result.get("best_index", 0))
        except Exception:
            idx = 0

        idx = max(0, min(idx, len(candidate_list) - 1))

        if scores:
            idx = max(range(len(scores)), key=lambda i: scores[i])

        return {
            "best_text": candidate_list[idx],
            "scores": [{"text": candidate_list[i], "score": scores[i]} for i in range(len(candidate_list))],
            "reason": str(result.get("reason", "")),
        }

    # =========================================================
    # fallback
    # =========================================================

    def _fallback_episode_evaluation(
        self,
        episode: Dict[str, Any],
        reason: str,
    ) -> Dict[str, Any]:
        input_tokens = [str(x) for x in episode.get("input_tokens", [])]
        mid_converged = [str(x) for x in episode.get("mid_converged", [])]
        final_expanded = [str(x) for x in episode.get("final_expanded", [])]
        response_text = str(episode.get("response_text", ""))

        input_size = max(1, len(input_tokens))
        mid_size = len(mid_converged)
        final_size = len(final_expanded)

        overlap = 0
        if response_text:
            for token in input_tokens:
                if token and token in response_text:
                    overlap += 1
        score_response = min(1.0, overlap / input_size + (0.15 if response_text else 0.0))

        if mid_size == 0:
            score_convergence = 0.15
        else:
            ratio = mid_size / input_size
            if ratio <= 1.5:
                score_convergence = 0.80
            elif ratio <= 3.0:
                score_convergence = 0.60
            else:
                score_convergence = 0.35

        if final_size >= input_size:
            score_divergence = 0.60
        elif final_size > 0:
            score_divergence = 0.45
        else:
            score_divergence = 0.20

        resp_len = len(response_text)
        if 4 <= resp_len <= 32:
            score_naturalness = 0.65
        elif resp_len > 0:
            score_naturalness = 0.45
        else:
            score_naturalness = 0.10

        complexity = mid_size + final_size
        if complexity <= 24:
            score_efficiency = 0.80
        elif complexity <= 80:
            score_efficiency = 0.60
        else:
            score_efficiency = 0.35

        score_total = (
            score_response * 0.30
            + score_divergence * 0.20
            + score_convergence * 0.20
            + score_efficiency * 0.15
            + score_naturalness * 0.15
        )

        return {
            "score_response": round(score_response, 6),
            "score_divergence": round(score_divergence, 6),
            "score_convergence": round(score_convergence, 6),
            "score_efficiency": round(score_efficiency, 6),
            "score_naturalness": round(score_naturalness, 6),
            "score_total": round(score_total, 6),
            "reason": reason,
        }

    def _fallback_verbal_candidates(
        self,
        input_tokens: List[str],
        final_tokens: List[str],
        candidates: List[str],
        reason: str,
    ) -> Dict[str, Any]:
        scored: List[Dict[str, Any]] = []

        for text in candidates:
            score = 0.0
            final_hit = sum(1 for t in final_tokens if t and t in text)
            input_hit = sum(1 for t in input_tokens if t and t in text)

            score += final_hit * 1.0
            score += input_hit * 0.2

            if text.endswith(("。", "！", "？", "…")):
                score += 0.3

            if 4 <= len(text) <= 32:
                score += 0.4
            elif len(text) <= 48:
                score += 0.2

            if "ですです" in text or "はは" in text or "もも" in text:
                score -= 0.5

            scored.append({"text": text, "score": round(score, 6)})

        scored.sort(key=lambda x: float(x["score"]), reverse=True)
        best_text = scored[0]["text"] if scored else (candidates[0] if candidates else "")

        return {
            "best_text": best_text,
            "scores": scored,
            "reason": reason,
        }
    def _fallback_recursive_state(
        self,
        original_input_tokens: List[str],
        current_input_tokens: List[str],
        raw_output_tokens: List[str],
        normalized_candidate_tokens: List[str],
        step_index: int,
        reason: str,
    ) -> Dict[str, Any]:
        original_set = set(t for t in original_input_tokens if t)
        raw_set = set(t for t in raw_output_tokens if t)
        norm_set = set(t for t in normalized_candidate_tokens if t)

        overlap_original = len(original_set & norm_set)
        overlap_current = len(set(current_input_tokens) & raw_set)

        score_total = 0.0
        if original_set:
            score_total += min(0.5, overlap_original / max(1, len(original_set)) * 0.5)
        if current_input_tokens:
            score_total += min(0.3, overlap_current / max(1, len(set(current_input_tokens))) * 0.3)
        if 1 <= len(normalized_candidate_tokens) <= 24:
            score_total += 0.2

        score_total = self._clip01(score_total)
        accepted = score_total >= 0.85

        return {
            "accepted": accepted,
            "score_total": round(score_total, 6),
            "reason": reason,
            "step": step_index,
        }