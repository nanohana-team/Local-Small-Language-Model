from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.apps.chat_control import ChatController
from src.core.primitive.convergence import ConvergenceModel
from src.core.primitive.divergence import DivergenceModel, DivergencePrimitive
from src.llm.evaluator_gemini import GeminiEvaluator
from src.llm.input_generator_gemini import GeminiInputGenerator
from src.utils.trainer import Trainer
from src.utils.verbal import (
    generate_candidates,
    load_verbal_model,
    score_candidates_locally,
    update_model_from_feedback,
)


class ChatLearningCentral:
    """
    学習の中央制御。
    - 入力は外から受け取る
    - trainer に反復思考をさせる
    - verbal 候補を評価して best_text を決める
    - 結果を保存する
    """

    def __init__(
        self,
        chat_controller: ChatController,
        trainer: Trainer,
        evaluator: Optional[Any] = None,
        input_generator: Optional[GeminiInputGenerator] = None,
        verbose: bool = False,
        save_path: str = "runtime/logs/chat_learning.jsonl",
        verbal_model_path: str = "runtime/models/verbal_model.json",
    ) -> None:
        self.chat_controller = chat_controller
        self.trainer = trainer
        self.evaluator = evaluator
        self.input_generator = input_generator
        self.verbose = verbose
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.verbal_model_path = verbal_model_path

    def _log(self, message: str, force: bool = False) -> None:
        if force or self.verbose:
            print(message, flush=True)

    @staticmethod
    def _preview_tokens(tokens: Sequence[str], limit: int = 20) -> str:
        if not tokens:
            return "[]"
        shown = list(tokens[:limit])
        suffix = "" if len(tokens) <= limit else f" ... (+{len(tokens) - limit})"
        return json.dumps(shown, ensure_ascii=False) + suffix

    def generate_input_tokens(self) -> List[str]:
        if self.input_generator is None:
            raise RuntimeError("input_generator is not set")
        tokens = self.input_generator.generate()
        return [str(x).strip() for x in tokens if str(x).strip()]

    def run_once(self, input_tokens: Sequence[str], depth: int = 4) -> Dict[str, Any]:
        started = time.time()
        episode_id = f"chat_ep_{uuid.uuid4().hex[:12]}"

        input_tokens = [str(x).strip() for x in input_tokens if str(x).strip()]
        if not input_tokens:
            raise ValueError("input_tokens is empty")

        self._log("=" * 88, force=True)
        self._log(f"[CHAT_CENTRAL] START id={episode_id} depth={depth}", force=True)
        self._log(
            f"[CHAT_CENTRAL] INPUT count={len(input_tokens)} tokens={self._preview_tokens(input_tokens, 40)}",
            force=True,
        )

        recursive_episode = self.trainer.run_recursive_episode(
            input_tokens=input_tokens,
            chat_controller=self.chat_controller,
            evaluator=self.evaluator,
            depth=depth,
        )

        final_output_tokens = list(recursive_episode.get("final_output_tokens", []))
        self._log(
            f"[CHAT_CENTRAL] FINAL_OUTPUT count={len(final_output_tokens)} tokens={self._preview_tokens(final_output_tokens, 40)}",
            force=True,
        )

        candidates = generate_candidates(
            final_tokens=final_output_tokens,
            original_input=list(input_tokens),
            limit=6,
            model_path=self.verbal_model_path,
        )
        self._log(
            f"[CHAT_CENTRAL] CANDIDATES count={len(candidates)}",
            force=True,
        )
        for i, c in enumerate(candidates, start=1):
            self._log(f"  [CANDIDATE {i}] {json.dumps(c, ensure_ascii=False)}", force=True)

        evaluation = self._evaluate_candidates(
            input_tokens=list(input_tokens),
            final_tokens=final_output_tokens,
            candidates=candidates,
        )

        best_text = str(evaluation.get("best_text", candidates[0] if candidates else "……"))
        self._log(
            f"[CHAT_CENTRAL] BEST {json.dumps(best_text, ensure_ascii=False)}",
            force=True,
        )

        model_after = update_model_from_feedback(
            chosen_text=best_text,
            candidates=candidates,
            final_tokens=final_output_tokens,
            original_input=list(input_tokens),
            model_path=self.verbal_model_path,
        )

        ended = time.time()
        result = {
            "episode_id": episode_id,
            "input_tokens": list(input_tokens),
            "recursive_episode": recursive_episode,
            "final_output_tokens": final_output_tokens,
            "candidates": list(candidates),
            "evaluation": evaluation,
            "best_text": best_text,
            "verbal_model": model_after,
            "timestamps": {
                "started_at_unix": started,
                "ended_at_unix": ended,
            },
            "metrics": {
                "elapsed_sec": round(ended - started, 6),
                "candidate_count": len(candidates),
                "input_token_count": len(input_tokens),
                "final_output_count": len(final_output_tokens),
                "recursive_step_count": int(recursive_episode.get("recursive_step_count", 0)),
            },
        }

        self._save_result(result)

        self._log(
            f"[CHAT_CENTRAL] END id={episode_id} elapsed={result['metrics']['elapsed_sec']:.3f}s",
            force=True,
        )
        return result

    def _evaluate_candidates(
        self,
        input_tokens: List[str],
        final_tokens: List[str],
        candidates: List[str],
    ) -> Dict[str, Any]:
        if not candidates:
            return {
                "best_text": "",
                "scores": [],
                "reason": "no_candidates",
            }

        method = getattr(self.evaluator, "evaluate_verbal_candidates", None)
        if callable(method):
            try:
                result = method(
                    input_tokens=input_tokens,
                    final_tokens=final_tokens,
                    candidates=candidates,
                )
                if isinstance(result, dict):
                    best_text = str(result.get("best_text", candidates[0]))
                    scores = result.get("scores", [])
                    return {
                        "best_text": best_text,
                        "scores": scores,
                        "reason": "external_evaluator",
                    }
            except Exception as exc:
                self._log(f"[WARN][CHAT_CENTRAL][EVAL] fallback reason={exc}", force=True)

        model = load_verbal_model(self.verbal_model_path)
        local_scores = score_candidates_locally(
            candidates=candidates,
            final_tokens=final_tokens,
            original_input=input_tokens,
            model=model,
        )
        local_scores.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

        return {
            "best_text": local_scores[0]["text"] if local_scores else candidates[0],
            "scores": local_scores,
            "reason": "local_fallback",
        }

    def _save_result(self, result: Dict[str, Any]) -> None:
        with self.save_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def build_chat_learning_central(args: Any) -> ChatLearningCentral:
    lexicon_path = Path(args.lexicon)
    div_model_path = Path(args.divergence_model)
    conv_model_path = Path(args.convergence_model)

    div_model_path.parent.mkdir(parents=True, exist_ok=True)
    conv_model_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[LEXICON] loading from {lexicon_path}", flush=True)
    lexicon = DivergencePrimitive.load_lexicon(lexicon_path)
    print(f"[LEXICON] loaded entries={len(lexicon)}", flush=True)

    divergence_model = DivergenceModel(
        lexicon=lexicon,
        model_path=div_model_path,
        default_branch=args.branch,
        final_branch=args.final_branch,
    )

    convergence_model = ConvergenceModel(
        divergence_model=divergence_model,
        model_path=conv_model_path,
    )

    evaluator = GeminiEvaluator(
        model_name=args.gemini_model,
        enabled=not args.disable_evaluator,
    )

    trainer = Trainer(
        divergence_model_path=div_model_path,
        convergence_model_path=conv_model_path,
        dict_path=Path(args.lexicon) if str(args.lexicon).lower().endswith(".json") else None,
        enable_unknown_token_expansion=False,
        gemini_model_name=args.gemini_model,
        max_recursive_steps=args.max_recursive_steps,
        accept_score_threshold=args.accept_score_threshold,
    )

    controller = ChatController(
        divergence_model=divergence_model,
        convergence_model=convergence_model,
        evaluator=evaluator,
        storage=None,
        trainer=trainer,
        verbose=args.verbose,
    )

    input_generator = GeminiInputGenerator(
        model_name=args.input_generator_model,
        enabled=args.auto_input,
    )

    return ChatLearningCentral(
        chat_controller=controller,
        trainer=trainer,
        evaluator=evaluator,
        input_generator=input_generator,
        verbose=args.verbose,
        save_path=args.save_path,
        verbal_model_path=args.verbal_model_path,
    )