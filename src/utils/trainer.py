from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.core.primitive.divergence import DivergenceModel
from src.core.primitive.convergence import ConvergenceModel
from src.utils.unknown_token_expander import UnknownTokenExpander


class Trainer:
    def __init__(
        self,
        divergence_model_path: Path,
        convergence_model_path: Path,
        dict_path: Optional[Path] = None,
        enable_unknown_token_expansion: bool = False,
        gemini_model_name: str = "gemini-2.5-flash-lite",
        max_recursive_steps: int = 6,
        accept_score_threshold: float = 8.5,
    ) -> None:
        self.divergence_model_path = Path(divergence_model_path)
        self.convergence_model_path = Path(convergence_model_path)
        self.max_recursive_steps = max(1, int(max_recursive_steps))
        self.accept_score_threshold = float(accept_score_threshold)

        self.unknown_token_expander: Optional[UnknownTokenExpander] = None

        safe_enable_unknown = (
            enable_unknown_token_expansion
            and dict_path is not None
            and Path(dict_path).suffix.lower() == ".json"
        )

        if safe_enable_unknown:
            self.unknown_token_expander = UnknownTokenExpander(
                dict_path=Path(dict_path),
                model_name=gemini_model_name,
                enabled=True,
            )
        elif enable_unknown_token_expansion:
            print(
                f"[TRAINER] unknown token expansion disabled: "
                f"dict_path must be JSON, got {dict_path}"
            )

    def _filter_trainable_episodes(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for ep in episodes:
            if not isinstance(ep, dict):
                continue
            if not ep.get("input_tokens"):
                continue
            out.append(ep)
        return out

    def update_models(
        self,
        episodes: List[Dict[str, Any]],
        divergence_model: DivergenceModel,
        convergence_model: ConvergenceModel,
    ) -> None:
        trainable = self._filter_trainable_episodes(episodes)
        if not trainable:
            print("[TRAINER] no trainable episodes")
            return

        if self.unknown_token_expander is not None:
            try:
                normalized_for_unknown: List[Dict[str, Any]] = []
                for ep in trainable:
                    normalized_for_unknown.append({
                        "input_tokens": ep.get("input_tokens", []),
                        "converged_tokens": ep.get("thought_converged", ep.get("mid_converged", [])),
                        "final_tokens": ep.get("final_output_tokens", ep.get("final_expanded", [])),
                        "output_tokens": ep.get("final_output_tokens", ep.get("final_expanded", [])),
                    })
                added = self.unknown_token_expander.expand_from_episodes(normalized_for_unknown)
                print(f"[TRAINER] unknown token expansion added={added}")
            except Exception as e:
                print(f"[TRAINER] unknown token expansion failed: {e}")

        divergence_model.update_from_episodes(trainable)
        convergence_model.update_from_episodes(trainable)

        divergence_model.save(self.divergence_model_path)
        convergence_model.save(self.convergence_model_path)

        avg_score = self._average_score(trainable)
        print(
            f"[TRAINER] updated divergence+convergence "
            f"episodes={len(trainable)} avg_score={avg_score:.2f}"
        )

    @staticmethod
    def _average_score(episodes: List[Dict[str, Any]]) -> float:
        scores: List[float] = []
        for ep in episodes:
            evaluation = ep.get("evaluation") or {}
            try:
                scores.append(float(evaluation.get("score_total", 0.0)))
            except Exception:
                continue
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    @staticmethod
    def _to_clean_tokens(tokens: Sequence[Any]) -> List[str]:
        out: List[str] = []
        for x in tokens:
            s = str(x).strip()
            if s:
                out.append(s)
        return out

    @staticmethod
    def _unique_keep_order(tokens: Sequence[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for t in tokens:
            if t in seen:
                continue
            seen.add(t)
            out.append(t)
        return out

    def _extract_raw_state_tokens(self, thought: Dict[str, Any]) -> List[str]:
        for key in ("raw_state_tokens", "thought_core", "thought_converged", "final_converged"):
            value = thought.get(key)
            if isinstance(value, list) and value:
                return self._to_clean_tokens(value)
        return []

    def _normalize_output_tokens(
        self,
        collected_stage_outputs: List[List[str]],
        final_thought: Dict[str, Any],
    ) -> List[str]:
        merged: List[str] = []

        final_converged = self._to_clean_tokens(final_thought.get("final_converged", []))
        merged.extend(final_converged)

        for stage in collected_stage_outputs:
            merged.extend(self._to_clean_tokens(stage))

        merged = self._unique_keep_order(merged)
        return merged

    def _evaluate_recursive_step(
        self,
        evaluator: Any,
        original_input_tokens: List[str],
        current_input_tokens: List[str],
        raw_output_tokens: List[str],
        normalized_candidate_tokens: List[str],
        step_index: int,
    ) -> Dict[str, Any]:
        if evaluator is None:
            return {
                "accepted": False,
                "score_total": 0.0,
                "reason": "no_evaluator",
                "step": step_index,
            }

        method = getattr(evaluator, "evaluate_recursive_state", None)
        if callable(method):
            try:
                result = method(
                    original_input_tokens=original_input_tokens,
                    current_input_tokens=current_input_tokens,
                    raw_output_tokens=raw_output_tokens,
                    normalized_candidate_tokens=normalized_candidate_tokens,
                    step_index=step_index,
                )
                if isinstance(result, dict):
                    score_total = float(result.get("score_total", 0.0))
                    accepted = bool(
                        result.get("accepted", False)
                        or score_total >= self.accept_score_threshold
                    )
                    result["score_total"] = score_total
                    result["accepted"] = accepted
                    result["step"] = step_index
                    return result
            except Exception as e:
                return {
                    "accepted": False,
                    "score_total": 0.0,
                    "reason": f"evaluate_recursive_state_failed: {e}",
                    "step": step_index,
                }

        return {
            "accepted": False,
            "score_total": 0.0,
            "reason": "no_recursive_eval_method",
            "step": step_index,
        }

    def run_recursive_episode(
        self,
        *,
        input_tokens: Sequence[str],
        chat_controller: Any,
        evaluator: Optional[Any] = None,
        depth: int = 4,
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        original_input_tokens = self._to_clean_tokens(input_tokens)
        if not original_input_tokens:
            raise ValueError("input_tokens is empty")

        recursive_steps = self.max_recursive_steps if max_steps is None else max(1, int(max_steps))

        current_input_tokens = list(original_input_tokens)
        history: List[Dict[str, Any]] = []
        collected_raw_outputs: List[List[str]] = []
        collected_normalized_outputs: List[List[str]] = []

        final_thought: Dict[str, Any] = {}
        final_evaluation: Dict[str, Any] = {
            "accepted": False,
            "score_total": 0.0,
            "reason": "not_started",
        }

        for step_index in range(1, recursive_steps + 1):
            print(f"[TRAINER][RECURSIVE] step={step_index} input={current_input_tokens}", flush=True)

            thought = chat_controller.think_once(
                input_tokens=list(current_input_tokens),
                depth=depth,
            )
            final_thought = thought

            raw_output_tokens = self._extract_raw_state_tokens(thought)
            normalized_output_tokens = self._to_clean_tokens(thought.get("final_converged", []))

            if not raw_output_tokens:
                raw_output_tokens = normalized_output_tokens[:]

            if not normalized_output_tokens:
                normalized_output_tokens = raw_output_tokens[:]

            collected_raw_outputs.append(raw_output_tokens)
            collected_normalized_outputs.append(normalized_output_tokens)

            evaluation = self._evaluate_recursive_step(
                evaluator=evaluator,
                original_input_tokens=original_input_tokens,
                current_input_tokens=current_input_tokens,
                raw_output_tokens=raw_output_tokens,
                normalized_candidate_tokens=normalized_output_tokens,
                step_index=step_index,
            )
            final_evaluation = evaluation

            history.append({
                "step": step_index,
                "input_tokens": list(current_input_tokens),
                "thought_core": self._to_clean_tokens(thought.get("thought_core", [])),
                "thought_converged": self._to_clean_tokens(thought.get("thought_converged", [])),
                "raw_output_tokens": list(raw_output_tokens),
                "normalized_output_tokens": list(normalized_output_tokens),
                "evaluation": evaluation,
            })

            print(
                "[TRAINER][RECURSIVE] "
                f"step={step_index} accepted={evaluation.get('accepted')} "
                f"score={evaluation.get('score_total', 0.0)} "
                f"raw={raw_output_tokens} normalized={normalized_output_tokens}",
                flush=True,
            )

            if bool(evaluation.get("accepted", False)):
                print(f"[TRAINER][RECURSIVE] accepted at step={step_index}", flush=True)
                break

            current_input_tokens = list(raw_output_tokens)

        final_output_tokens = self._normalize_output_tokens(
            collected_stage_outputs=collected_normalized_outputs,
            final_thought=final_thought,
        )

        episode = {
            "input_tokens": list(original_input_tokens),
            "recursive_history": history,
            "thought_core": self._to_clean_tokens(final_thought.get("thought_core", [])),
            "thought_converged": self._to_clean_tokens(final_thought.get("thought_converged", [])),
            "final_converged": self._to_clean_tokens(final_thought.get("final_converged", [])),
            "all_raw_outputs": collected_raw_outputs,
            "all_normalized_outputs": collected_normalized_outputs,
            "final_output_tokens": final_output_tokens,
            "evaluation": final_evaluation,
            "accepted": bool(final_evaluation.get("accepted", False)),
            "recursive_step_count": len(history),
        }

        return episode