# src/utils/trainer.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.core.primitive.divergence import DivergenceModel
from src.core.primitive.convergence import ConvergenceModel


class Trainer:
    def __init__(
        self,
        divergence_model_path: Path,
        convergence_model_path: Path,
    ) -> None:
        self.divergence_model_path = Path(divergence_model_path)
        self.convergence_model_path = Path(convergence_model_path)

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