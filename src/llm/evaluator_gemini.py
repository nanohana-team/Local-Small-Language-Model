from __future__ import annotations

import re
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

    def _simple_text_tokens(self, text: str) -> List[str]:
        parts = re.split(r"[\s、。！？…,.!?\(\)\[\]「」『』]+", text)
        return [x for x in parts if x]

    def _token_overlap_ratio(self, base_tokens: Sequence[str], target_tokens: Sequence[str]) -> float:
        base = set(str(x) for x in base_tokens if str(x).strip())
        target = set(str(x) for x in target_tokens if str(x).strip())
        if not base:
            return 0.0
        return len(base & target) / len(base)

    def _noise_ratio(self, reference_tokens: Sequence[str], target_tokens: Sequence[str]) -> float:
        ref = set(str(x) for x in reference_tokens if str(x).strip())
        tgt = [str(x) for x in target_tokens if str(x).strip()]
        if not tgt:
            return 1.0
        noise = sum(1 for t in tgt if t not in ref)
        return noise / len(tgt)

    def _contains_sentence_ending(self, text: str) -> bool:
        return text.endswith(("。", "！", "？", "…"))

    def _is_question_like(self, input_tokens: Sequence[str], text: str) -> bool:
        joined = "".join(str(x) for x in input_tokens)
        return ("？" in joined or "か" in joined or "ですか" in joined) and (
            "？" in text or text.endswith("か") or "ですか" in text
        )

    def _looks_ungrammatical(self, text: str) -> bool:
        bad_patterns = [
            "ですです",
            "ますますます",
            "もしもし上",
            "教室ごめん出口",
            "深夜うん",
            "悲しいすれば",
            "不安いる",
            "平気予定",
        ]
        if any(p in text for p in bad_patterns):
            return True

        if len(text) >= 6 and "。" not in text and "？" not in text and "！" not in text:
            return True

        return False

    def _count_input_tokens_in_text(self, input_tokens: Sequence[str], text: str) -> int:
        count = 0
        for tok in input_tokens:
            s = str(tok).strip()
            if not s or s in {"。", "、"}:
                continue
            if s in text:
                count += 1
        return count

    def _has_basic_japanese_structure(self, text: str) -> bool:
        particles = ("は", "が", "を", "に", "で", "の", "と", "から", "まで", "へ")
        endings = ("です", "ます", "でした", "ません", "たい", "だ", "する", "した", "いる", "ある")
        return any(p in text for p in particles) or any(e in text for e in endings)

    def _normalize_text(self, text: str) -> str:
        return str(text or "").strip().replace(" ", "").replace("　", "")

    def _char_jaccard(self, a: str, b: str) -> float:
        sa = set(self._normalize_text(a))
        sb = set(self._normalize_text(b))
        if not sa and not sb:
            return 1.0
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / max(1, len(sa | sb))

    def _token_f1(self, predicted: Sequence[str], target: Sequence[str]) -> float:
        pred = [str(x).strip() for x in predicted if str(x).strip()]
        gold = [str(x).strip() for x in target if str(x).strip()]
        if not pred and not gold:
            return 1.0
        if not pred or not gold:
            return 0.0

        pred_set = set(pred)
        gold_set = set(gold)

        tp = len(pred_set & gold_set)
        if tp <= 0:
            return 0.0

        precision = tp / max(1, len(pred_set))
        recall = tp / max(1, len(gold_set))
        if precision + recall <= 0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    def evaluate_episode(self, episode: Dict[str, Any]) -> Dict[str, Any]:
        heuristic = self._fallback_episode_evaluation(episode=episode, reason="heuristic")

        if not self.enabled:
            heuristic["reason"] = "disabled"
            return heuristic

        target_tokens = [str(x) for x in episode.get("target_tokens", []) if str(x).strip()]
        target_text = str(episode.get("target_text", "")).strip()

        prompt = {
            "instruction": (
                "Return ONLY valid JSON object. No explanation. No markdown. "
                "Evaluate this episode for supervised reinforcement learning. "
                "Be strict. Prioritize closeness to the teacher target over general plausibility. "
                "Penalize semantic drift, broken Japanese, and outputs that ignore the teacher target."
            ),
            "task": "evaluate_episode",
            "layer": "core_thought_supervised",
            "input_tokens": [str(x) for x in episode.get("input_tokens", [])],
            "target_tokens": target_tokens,
            "target_text": target_text,
            "initial_core": [str(x) for x in episode.get("initial_core", [])],
            "mid_converged": [str(x) for x in episode.get("mid_converged", [])],
            "final_expanded": [str(x) for x in episode.get("final_expanded", [])],
            "response_text": str(episode.get("response_text", "")),
            "criteria": [
                "teacher_alignment",
                "response_quality",
                "convergence_quality",
                "thought_efficiency",
                "naturalness",
            ],
            "output_format": {
                "score_teacher_alignment": 0.0,
                "score_response": 0.0,
                "score_convergence": 0.0,
                "score_efficiency": 0.0,
                "score_naturalness": 0.0,
                "reason": "",
            },
        }

        try:
            self._log(f"[EVAL][episode] router_call model={self.model_name or 'auto'}")
            result = self.router.generate_json(prompt, model_name=self.model_name)
            return self._normalize_episode_evaluation(result, heuristic)
        except Exception as exc:
            self._log(f"[EVAL][episode] fallback reason={type(exc).__name__}:{exc}")
            heuristic["reason"] = f"fallback:{type(exc).__name__}:{exc}"
            return heuristic

    def _normalize_episode_evaluation(
        self,
        result: Dict[str, Any],
        heuristic: Dict[str, Any],
    ) -> Dict[str, Any]:
        llm_teacher = self._clip01(
            self._to_float(result.get("score_teacher_alignment"), heuristic["score_teacher_alignment"])
        )
        llm_response = self._clip01(self._to_float(result.get("score_response"), heuristic["score_response"]))
        llm_convergence = self._clip01(
            self._to_float(result.get("score_convergence"), heuristic["score_convergence"])
        )
        llm_efficiency = self._clip01(self._to_float(result.get("score_efficiency"), heuristic["score_efficiency"]))
        llm_naturalness = self._clip01(
            self._to_float(result.get("score_naturalness"), heuristic["score_naturalness"])
        )

        score_teacher_alignment = min(
            llm_teacher,
            self._clip01(heuristic["score_teacher_alignment"] + 0.10),
        )
        score_response = min(llm_response, self._clip01(heuristic["score_response"] + 0.15))
        score_convergence = min(llm_convergence, self._clip01(heuristic["score_convergence"] + 0.15))
        score_efficiency = min(llm_efficiency, self._clip01(heuristic["score_efficiency"] + 0.20))
        score_naturalness = min(llm_naturalness, self._clip01(heuristic["score_naturalness"] + 0.15))

        score_total = (
            score_teacher_alignment * 0.55
            + score_response * 0.15
            + score_convergence * 0.10
            + score_efficiency * 0.05
            + score_naturalness * 0.15
        )

        return {
            "score_teacher_alignment": round(score_teacher_alignment, 6),
            "score_response": round(score_response, 6),
            "score_convergence": round(score_convergence, 6),
            "score_efficiency": round(score_efficiency, 6),
            "score_naturalness": round(score_naturalness, 6),
            "score_total": round(self._clip01(score_total), 6),
            "reason": str(result.get("reason", "")),
        }

    def evaluate_recursive_state(
        self,
        original_input_tokens: List[str],
        current_input_tokens: List[str],
        raw_output_tokens: List[str],
        normalized_candidate_tokens: List[str],
        step_index: int,
        target_tokens: Optional[List[str]] = None,
        target_text: str = "",
    ) -> Dict[str, Any]:
        heuristic = self._fallback_recursive_state(
            original_input_tokens=original_input_tokens,
            current_input_tokens=current_input_tokens,
            raw_output_tokens=raw_output_tokens,
            normalized_candidate_tokens=normalized_candidate_tokens,
            step_index=step_index,
            target_tokens=target_tokens or [],
            target_text=target_text,
            reason="heuristic",
        )

        if not self.enabled:
            heuristic["reason"] = "disabled"
            return heuristic

        prompt = {
            "instruction": (
                "Return ONLY valid JSON object. No explanation. No markdown. "
                "Evaluate whether this recursive thought step is good enough to stop in supervised reinforcement learning. "
                "Prioritize closeness to the teacher target. Penalize extra unrelated tokens."
            ),
            "task": "evaluate_recursive_state",
            "layer": "core_thought_recursive_supervised",
            "original_input_tokens": [str(x) for x in original_input_tokens],
            "current_input_tokens": [str(x) for x in current_input_tokens],
            "raw_output_tokens": [str(x) for x in raw_output_tokens],
            "normalized_candidate_tokens": [str(x) for x in normalized_candidate_tokens],
            "target_tokens": [str(x) for x in (target_tokens or [])],
            "target_text": str(target_text or ""),
            "step_index": int(step_index),
            "criteria": [
                "teacher_alignment",
                "meaning_preservation",
                "thought_progress",
                "stability",
                "stop_readiness",
            ],
            "output_format": {
                "score_total": 0.0,
                "reason": "",
            },
        }

        try:
            self._log(f"[EVAL][recursive_state] router_call model={self.model_name or 'auto'}")
            result = self.router.generate_json(prompt, model_name=self.model_name)
            llm_score = self._clip01(self._to_float(result.get("score_total"), heuristic["score_total"]))
            score_total = min(llm_score, self._clip01(heuristic["score_total"] + 0.10))

            return {
                "accepted": False,
                "score_total": round(score_total, 6),
                "reason": str(result.get("reason", "")),
                "step": step_index,
                "teacher_alignment": heuristic.get("teacher_alignment", 0.0),
                "preservation": heuristic.get("preservation", 0.0),
                "noise": heuristic.get("noise", 1.0),
                "target_text_similarity": heuristic.get("target_text_similarity", 0.0),
            }
        except Exception as exc:
            self._log(f"[EVAL][recursive_state] fallback reason={type(exc).__name__}:{exc}")
            heuristic["reason"] = f"fallback:{type(exc).__name__}:{exc}"
            return heuristic

    def evaluate_verbal_candidates(
        self,
        input_tokens: List[str],
        final_tokens: List[str],
        candidates: List[str],
        target_tokens: Optional[List[str]] = None,
        target_text: str = "",
    ) -> Dict[str, Any]:
        if not candidates:
            return {"best_text": "", "scores": [], "reason": "no_candidates"}

        target_tokens = [str(x).strip() for x in (target_tokens or []) if str(x).strip()]
        target_text = str(target_text or "").strip()

        local_result = self._fallback_verbal_candidates(
            input_tokens=input_tokens,
            final_tokens=final_tokens,
            candidates=candidates,
            target_tokens=target_tokens,
            target_text=target_text,
            reason="local_primary",
        )

        if not self.enabled:
            return local_result

        local_scores = local_result.get("scores", [])
        if not isinstance(local_scores, list) or not local_scores:
            return local_result

        sorted_local = sorted(
            [
                {"text": str(x.get("text", "")), "score": float(x.get("score", 0.0))}
                for x in local_scores
            ],
            key=lambda x: x["score"],
            reverse=True,
        )

        top_candidates = [x["text"] for x in sorted_local[: min(3, len(sorted_local))]]
        if len(top_candidates) <= 1:
            return local_result

        prompt = {
            "instruction": (
                "Return ONLY valid JSON object. No explanation. No markdown. "
                "Choose the best Japanese sentence candidate among the shortlisted candidates. "
                "Prefer the candidate closest to the teacher target, then naturalness, then fluency. "
                "Penalize unrelated extra words."
            ),
            "task": "evaluate_text_candidates",
            "layer": "verbal_surface_shortlist_supervised",
            "input_tokens": [str(x) for x in input_tokens],
            "thought_tokens": [str(x) for x in final_tokens],
            "target_tokens": [str(x) for x in target_tokens],
            "target_text": target_text,
            "candidates": top_candidates,
            "criteria": [
                "teacher_alignment",
                "meaning_preservation",
                "naturalness",
                "fluency",
                "noise_avoidance",
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
            llm_result = self._normalize_verbal_candidate_evaluation(result, top_candidates)

            local_best = str(local_result.get("best_text", ""))
            llm_best = str(llm_result.get("best_text", ""))

            local_score_map = {
                str(x.get("text", "")): float(x.get("score", 0.0))
                for x in sorted_local
            }

            local_best_score = float(local_score_map.get(local_best, 0.0))
            llm_best_score = float(local_score_map.get(llm_best, 0.0))

            if self._looks_ungrammatical(llm_best) and not self._looks_ungrammatical(local_best):
                return {
                    "best_text": local_best,
                    "scores": sorted_local,
                    "reason": "local_override_ungrammatical_llm_choice",
                }

            if llm_best_score + 1.0 < local_best_score:
                return {
                    "best_text": local_best,
                    "scores": sorted_local,
                    "reason": "local_override_large_score_gap",
                }

            if llm_best and llm_best_score >= local_best_score - 0.6:
                return {
                    "best_text": llm_best,
                    "scores": sorted_local,
                    "reason": "local_primary_llm_refined",
                }

            return {
                "best_text": local_best,
                "scores": sorted_local,
                "reason": "local_primary_with_llm_checked",
            }
        except Exception as exc:
            self._log(f"[EVAL][verbal_candidates] fallback reason={type(exc).__name__}:{exc}")
            local_result["reason"] = f"fallback:{type(exc).__name__}:{exc}"
            return local_result

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

    def _fallback_episode_evaluation(
        self,
        episode: Dict[str, Any],
        reason: str,
    ) -> Dict[str, Any]:
        input_tokens = [str(x) for x in episode.get("input_tokens", []) if str(x).strip()]
        target_tokens = [str(x) for x in episode.get("target_tokens", []) if str(x).strip()]
        target_text = str(episode.get("target_text", "")).strip()

        mid_converged = [str(x) for x in episode.get("mid_converged", []) if str(x).strip()]
        final_expanded = [str(x) for x in episode.get("final_expanded", []) if str(x).strip()]
        response_text = str(episode.get("response_text", "")).strip()

        response_tokens = self._simple_text_tokens(response_text)

        preserve_response = self._token_overlap_ratio(input_tokens, response_tokens)
        preserve_mid = self._token_overlap_ratio(input_tokens, mid_converged)
        preserve_final = self._token_overlap_ratio(input_tokens, final_expanded)

        response_noise = self._noise_ratio(input_tokens, response_tokens)
        final_noise = self._noise_ratio(input_tokens, final_expanded)

        teacher_alignment_tokens = 0.0
        teacher_alignment_text = 0.0

        if target_tokens:
            teacher_alignment_tokens = max(
                self._token_f1(final_expanded, target_tokens),
                self._token_f1(response_tokens, target_tokens),
                self._token_f1(mid_converged, target_tokens),
            )
        if target_text:
            teacher_alignment_text = max(
                self._char_jaccard(response_text, target_text),
                self._char_jaccard("".join(final_expanded), target_text),
            )

        score_teacher_alignment = self._clip01(
            teacher_alignment_tokens * 0.75 + teacher_alignment_text * 0.25
        )

        score_response = self._clip01(
            teacher_alignment_tokens * 0.45
            + teacher_alignment_text * 0.20
            + preserve_response * 0.20
            + (0.10 if response_text else 0.0)
            - response_noise * 0.20
        )

        score_convergence = self._clip01(
            teacher_alignment_tokens * 0.50
            + preserve_mid * 0.20
            + preserve_final * 0.15
            - final_noise * 0.15
        )

        score_efficiency = 0.75
        complexity = len(mid_converged) + len(final_expanded)
        if complexity > 48:
            score_efficiency = 0.45
        elif complexity > 28:
            score_efficiency = 0.60

        score_naturalness = 0.20
        if response_text:
            if self._contains_sentence_ending(response_text):
                score_naturalness += 0.20
            if 4 <= len(response_text) <= 32:
                score_naturalness += 0.20
            if self._is_question_like(input_tokens, response_text):
                score_naturalness += 0.10
            if not self._looks_ungrammatical(response_text):
                score_naturalness += 0.20
        score_naturalness = self._clip01(score_naturalness)

        score_total = (
            score_teacher_alignment * 0.55
            + score_response * 0.15
            + score_convergence * 0.10
            + score_efficiency * 0.05
            + score_naturalness * 0.15
        )

        return {
            "score_teacher_alignment": round(score_teacher_alignment, 6),
            "score_response": round(score_response, 6),
            "score_convergence": round(score_convergence, 6),
            "score_efficiency": round(score_efficiency, 6),
            "score_naturalness": round(score_naturalness, 6),
            "score_total": round(self._clip01(score_total), 6),
            "reason": reason,
        }

    def _fallback_verbal_candidates(
        self,
        input_tokens: List[str],
        final_tokens: List[str],
        candidates: List[str],
        target_tokens: Optional[List[str]],
        target_text: str,
        reason: str,
    ) -> Dict[str, Any]:
        input_set = set(str(x) for x in input_tokens if str(x).strip())
        final_set = set(str(x) for x in final_tokens if str(x).strip())
        target_set = set(str(x) for x in (target_tokens or []) if str(x).strip())
        input_joined = "".join(str(x) for x in input_tokens if str(x).strip())

        scored: List[Dict[str, Any]] = []

        for text in candidates:
            text_tokens = self._simple_text_tokens(text)
            text_set = set(text_tokens)

            input_hit = len(input_set & text_set)
            final_hit = len(final_set & text_set)
            target_hit = len(target_set & text_set)

            noise = len(text_set - input_set - final_set - target_set)
            exact_input_hit = self._count_input_tokens_in_text(input_tokens, text)

            score = 0.0

            score += input_hit * 1.2
            score += exact_input_hit * 0.4
            score += final_hit * 0.4

            if target_set:
                score += target_hit * 3.2

            if target_text:
                score += self._char_jaccard(text, target_text) * 4.0

            score -= noise * 1.6

            if 4 <= len(text) <= 36:
                score += 0.4
            elif len(text) > 48:
                score -= 0.6

            if text.endswith(("。", "！", "？", "…")):
                score += 0.5

            if self._is_question_like(input_tokens, text):
                score += 0.5

            if any(p in text for p in ("は", "が", "を", "に", "で", "の", "と", "から", "まで", "へ")):
                score += 0.6
            if any(e in text for e in ("です", "ます", "でした", "ません", "たい", "だ", "する", "した", "いる", "ある")):
                score += 0.5

            if all(tok in text for tok in input_tokens if str(tok).strip() and tok not in ("。", "、")):
                score += 1.0

            if text == input_joined or text == input_joined.replace("？", "。"):
                score += 1.0

            if self._has_basic_japanese_structure(text):
                score += 0.4

            if self._looks_ungrammatical(text):
                score -= 2.2

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
        target_tokens: List[str],
        target_text: str,
        reason: str,
    ) -> Dict[str, Any]:
        preservation = self._token_overlap_ratio(original_input_tokens, normalized_candidate_tokens)
        stability = self._token_overlap_ratio(current_input_tokens, raw_output_tokens)
        noise = self._noise_ratio(original_input_tokens, normalized_candidate_tokens)

        teacher_alignment = 0.0
        if target_tokens:
            teacher_alignment = self._token_f1(normalized_candidate_tokens, target_tokens)

        target_text_similarity = 0.0
        if target_text:
            target_text_similarity = self._char_jaccard("".join(normalized_candidate_tokens), target_text)

        length_bonus = 0.10 if 1 <= len(normalized_candidate_tokens) <= max(12, len(original_input_tokens) + 4) else 0.0

        score_total = (
            teacher_alignment * 0.60
            + target_text_similarity * 0.15
            + preservation * 0.10
            + stability * 0.05
            + length_bonus
            - min(0.35, noise * 0.35)
        )
        score_total = self._clip01(score_total)

        return {
            "accepted": False,
            "score_total": round(score_total, 6),
            "reason": reason,
            "step": step_index,
            "teacher_alignment": round(teacher_alignment, 6),
            "preservation": round(preservation, 6),
            "noise": round(noise, 6),
            "target_text_similarity": round(target_text_similarity, 6),
        }