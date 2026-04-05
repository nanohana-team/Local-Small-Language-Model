from __future__ import annotations

import copy
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
        accept_score_threshold: float = 0.78,
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

    @staticmethod
    def _clip01(value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

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

    @staticmethod
    def _token_overlap_ratio(base_tokens: Sequence[str], target_tokens: Sequence[str]) -> float:
        base = set(t for t in base_tokens if t)
        target = set(t for t in target_tokens if t)
        if not base:
            return 0.0
        return len(base & target) / len(base)

    @staticmethod
    def _noise_ratio(reference_tokens: Sequence[str], target_tokens: Sequence[str]) -> float:
        ref = set(t for t in reference_tokens if t)
        tgt = [t for t in target_tokens if t]
        if not tgt:
            return 1.0
        noise = sum(1 for t in tgt if t not in ref)
        return noise / len(tgt)

    @staticmethod
    def _char_jaccard(a: str, b: str) -> float:
        sa = set(str(a or "").strip().replace(" ", "").replace("　", ""))
        sb = set(str(b or "").strip().replace(" ", "").replace("　", ""))
        if not sa and not sb:
            return 1.0
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / max(1, len(sa | sb))

    @staticmethod
    def _token_f1(predicted: Sequence[str], target: Sequence[str]) -> float:
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

    def _filter_trainable_episodes(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for ep in episodes:
            if not isinstance(ep, dict):
                continue
            if not ep.get("input_tokens"):
                continue
            out.append(ep)
        return out

    def _shape_training_score(self, raw_score: float) -> float:
        """
        学習用報酬整形。
        - 低品質(～0.35): 低いまま保つ
        - 中品質(0.35～): そこそこ良い回が全部「罰」側に寄りにくいよう少し持ち上げる
        """
        raw = self._clip01(float(raw_score))

        if raw < 0.35:
            shaped = raw * 0.9
        else:
            shaped = 0.60 + ((raw - 0.35) / 0.65) * 0.40

        return round(self._clip01(shaped), 6)

    def _build_supervised_episode_score(self, episode: Dict[str, Any]) -> float:
        evaluation = episode.get("evaluation") or {}
        try:
            base_score = float(evaluation.get("score_total", 0.0))
        except Exception:
            base_score = 0.0

        target_tokens = self._to_clean_tokens(episode.get("target_tokens", []))
        target_text = str(episode.get("target_text", "")).strip()

        final_tokens = self._to_clean_tokens(
            episode.get("final_output_tokens", episode.get("final_expanded", []))
        )
        response_text = str(episode.get("best_text", "") or episode.get("response_text", "")).strip()

        token_match = self._token_f1(final_tokens, target_tokens) if target_tokens else 0.0
        text_match = self._char_jaccard(response_text, target_text) if target_text else 0.0
        noise = self._noise_ratio(target_tokens or episode.get("input_tokens", []), final_tokens)

        supervised_score = (
            token_match * 0.60
            + text_match * 0.25
            + base_score * 0.15
            - min(0.20, noise * 0.20)
        )
        return round(self._clip01(supervised_score), 6)

    def _prepare_training_episodes(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prepared: List[Dict[str, Any]] = []

        for ep in episodes:
            cloned = copy.deepcopy(ep)
            evaluation = cloned.setdefault("evaluation", {})

            try:
                raw_score = float(evaluation.get("score_total", 0.0))
            except Exception:
                raw_score = 0.0

            has_teacher = bool(cloned.get("target_tokens") or cloned.get("target_text"))

            if has_teacher:
                supervised_score = self._build_supervised_episode_score(cloned)
                evaluation["score_total_raw"] = round(raw_score, 6)
                evaluation["score_total_supervised"] = supervised_score
                evaluation["score_total"] = supervised_score
                evaluation["score_total_train"] = supervised_score
            else:
                shaped_score = self._shape_training_score(raw_score)
                evaluation["score_total_raw"] = round(raw_score, 6)
                evaluation["score_total"] = shaped_score
                evaluation["score_total_train"] = shaped_score

            prepared.append(cloned)

        return prepared

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

        trainable_for_update = self._prepare_training_episodes(trainable)

        raw_avg = self._average_score(trainable)
        shaped_avg = self._average_score(trainable_for_update)
        print(
            f"[TRAINER] reward shaping raw_avg={raw_avg:.3f} -> train_avg={shaped_avg:.3f}",
            flush=True,
        )

        if self.unknown_token_expander is not None:
            try:
                normalized_for_unknown: List[Dict[str, Any]] = []
                for ep in trainable_for_update:
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

        divergence_model.update_from_episodes(trainable_for_update)
        convergence_model.update_from_episodes(trainable_for_update)

        divergence_model.save(self.divergence_model_path)
        convergence_model.save(self.convergence_model_path)

        print(
            f"[TRAINER] updated divergence+convergence "
            f"episodes={len(trainable_for_update)} raw_avg={raw_avg:.2f} train_avg={shaped_avg:.2f}"
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
        target_tokens: Optional[List[str]] = None,
        target_text: str = "",
    ) -> Dict[str, Any]:
        if evaluator is None:
            teacher_alignment = self._token_f1(normalized_candidate_tokens, target_tokens or [])
            text_similarity = self._char_jaccard("".join(normalized_candidate_tokens), target_text) if target_text else 0.0
            preservation = self._token_overlap_ratio(original_input_tokens, normalized_candidate_tokens)
            noise = self._noise_ratio(target_tokens or original_input_tokens, normalized_candidate_tokens)

            score_total = self._clip01(
                teacher_alignment * 0.60
                + text_similarity * 0.15
                + preservation * 0.15
                - min(0.30, noise * 0.25)
            )

            return {
                "accepted": False,
                "score_total": round(score_total, 6),
                "reason": "no_evaluator",
                "step": step_index,
                "teacher_alignment": round(teacher_alignment, 6),
                "preservation": round(preservation, 6),
                "noise": round(noise, 6),
                "target_text_similarity": round(text_similarity, 6),
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
                    target_tokens=target_tokens or [],
                    target_text=target_text,
                )
                if isinstance(result, dict):
                    score_total = self._clip01(float(result.get("score_total", 0.0)))

                    preservation = self._token_overlap_ratio(
                        original_input_tokens,
                        normalized_candidate_tokens,
                    )
                    teacher_alignment = self._token_f1(normalized_candidate_tokens, target_tokens or [])
                    text_similarity = self._char_jaccard("".join(normalized_candidate_tokens), target_text) if target_text else 0.0
                    noise = self._noise_ratio(
                        target_tokens or original_input_tokens,
                        normalized_candidate_tokens,
                    )

                    if target_tokens or target_text:
                        accepted = bool(
                            score_total >= self.accept_score_threshold
                            and teacher_alignment >= 0.72
                            and noise <= 0.30
                        )
                    else:
                        accepted = bool(
                            score_total >= self.accept_score_threshold
                            and preservation >= 0.55
                            and noise <= 0.45
                        )

                    result["score_total"] = round(score_total, 6)
                    result["accepted"] = accepted
                    result["step"] = step_index
                    result["teacher_alignment"] = round(teacher_alignment, 6)
                    result["preservation"] = round(preservation, 6)
                    result["noise"] = round(noise, 6)
                    result["target_text_similarity"] = round(text_similarity, 6)
                    return result
            except Exception as e:
                return {
                    "accepted": False,
                    "score_total": 0.0,
                    "reason": f"evaluate_recursive_state_failed: {e}",
                    "step": step_index,
                    "teacher_alignment": 0.0,
                    "preservation": 0.0,
                    "noise": 1.0,
                    "target_text_similarity": 0.0,
                }

        return {
            "accepted": False,
            "score_total": 0.0,
            "reason": "no_recursive_eval_method",
            "step": step_index,
            "teacher_alignment": 0.0,
            "preservation": 0.0,
            "noise": 1.0,
            "target_text_similarity": 0.0,
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
            "teacher_alignment": 0.0,
            "preservation": 0.0,
            "noise": 1.0,
            "target_text_similarity": 0.0,
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
                f"preservation={evaluation.get('preservation', 0.0)} "
                f"noise={evaluation.get('noise', 1.0)} "
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
            "target_tokens": [],
            "target_text": "",
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

    def run_supervised_recursive_episode(
        self,
        *,
        input_tokens: Sequence[str],
        chat_controller: Any,
        evaluator: Optional[Any] = None,
        depth: int = 4,
        target_tokens: Optional[Sequence[str]] = None,
        target_text: str = "",
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        original_input_tokens = self._to_clean_tokens(input_tokens)
        gold_tokens = self._to_clean_tokens(target_tokens or [])
        gold_text = str(target_text or "").strip()

        if not original_input_tokens:
            raise ValueError("input_tokens is empty")
        if not gold_tokens and not gold_text:
            return self.run_recursive_episode(
                input_tokens=original_input_tokens,
                chat_controller=chat_controller,
                evaluator=evaluator,
                depth=depth,
                max_steps=max_steps,
            )

        recursive_steps = self.max_recursive_steps if max_steps is None else max(1, int(max_steps))

        current_input_tokens = list(original_input_tokens)
        history: List[Dict[str, Any]] = []
        collected_raw_outputs: List[List[str]] = []
        collected_normalized_outputs: List[List[str]] = []

        best_step_payload: Optional[Dict[str, Any]] = None
        best_supervised_score = -1.0

        final_thought: Dict[str, Any] = {}
        final_evaluation: Dict[str, Any] = {
            "accepted": False,
            "score_total": 0.0,
            "reason": "not_started",
            "teacher_alignment": 0.0,
            "preservation": 0.0,
            "noise": 1.0,
            "target_text_similarity": 0.0,
        }

        for step_index in range(1, recursive_steps + 1):
            print(
                f"[TRAINER][SUPERVISED] step={step_index} input={current_input_tokens} "
                f"target_tokens={gold_tokens} target_text={target_text}",
                flush=True,
            )

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
                target_tokens=gold_tokens,
                target_text=gold_text,
            )
            final_evaluation = evaluation

            teacher_alignment = float(evaluation.get("teacher_alignment", 0.0))
            text_similarity = float(evaluation.get("target_text_similarity", 0.0))
            preservation = float(evaluation.get("preservation", 0.0))
            noise = float(evaluation.get("noise", 1.0))

            supervised_score = self._clip01(
                teacher_alignment * 0.60
                + text_similarity * 0.20
                + preservation * 0.10
                - min(0.20, noise * 0.10)
            )

            step_payload = {
                "step": step_index,
                "input_tokens": list(current_input_tokens),
                "thought_core": self._to_clean_tokens(thought.get("thought_core", [])),
                "thought_converged": self._to_clean_tokens(thought.get("thought_converged", [])),
                "raw_output_tokens": list(raw_output_tokens),
                "normalized_output_tokens": list(normalized_output_tokens),
                "evaluation": {
                    **evaluation,
                    "supervised_score": round(supervised_score, 6),
                },
            }
            history.append(step_payload)

            if supervised_score > best_supervised_score:
                best_supervised_score = supervised_score
                best_step_payload = {
                    "thought": thought,
                    "step_index": step_index,
                    "raw_output_tokens": list(raw_output_tokens),
                    "normalized_output_tokens": list(normalized_output_tokens),
                    "evaluation": {
                        **evaluation,
                        "supervised_score": round(supervised_score, 6),
                    },
                }

            print(
                "[TRAINER][SUPERVISED] "
                f"step={step_index} accepted={evaluation.get('accepted')} "
                f"score={evaluation.get('score_total', 0.0)} "
                f"teacher_alignment={teacher_alignment:.6f} "
                f"text_similarity={text_similarity:.6f} "
                f"preservation={preservation:.6f} "
                f"noise={noise:.6f} "
                f"supervised_score={supervised_score:.6f} "
                f"raw={raw_output_tokens} normalized={normalized_output_tokens}",
                flush=True,
            )

            if bool(evaluation.get("accepted", False)):
                print(f"[TRAINER][SUPERVISED] accepted at step={step_index}", flush=True)
                break

            current_input_tokens = list(raw_output_tokens)

        if best_step_payload is not None:
            final_thought = best_step_payload["thought"]
            final_evaluation = dict(best_step_payload["evaluation"])
            final_output_tokens = self._normalize_output_tokens(
                collected_stage_outputs=[best_step_payload["normalized_output_tokens"]],
                final_thought=final_thought,
            )
            selected_step = int(best_step_payload["step_index"])
        else:
            final_output_tokens = self._normalize_output_tokens(
                collected_stage_outputs=collected_normalized_outputs,
                final_thought=final_thought,
            )
            selected_step = len(history)

        episode = {
            "input_tokens": list(original_input_tokens),
            "target_tokens": list(gold_tokens),
            "target_text": gold_text,
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
            "selected_best_step": selected_step,
            "selected_best_supervised_score": round(best_supervised_score if best_supervised_score >= 0.0 else 0.0, 6),
        }

        return episode