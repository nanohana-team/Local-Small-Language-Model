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
from src.llm.input_output_generator_gemini import GeminiTeacherPairGenerator
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

    追加:
    - 教師あり学習用に target_tokens / target_text を受け取れる
    - 候補文評価でも教師を優先できる
    - auto_teacher 用に input/target ペアを自動生成できる
    """

    def __init__(
        self,
        chat_controller: ChatController,
        trainer: Trainer,
        evaluator: Optional[Any] = None,
        teacher_generator: Optional[Any] = None,
        verbose: bool = False,
        save_path: str = "runtime/logs/chat_learning.jsonl",
        verbal_model_path: str = "runtime/models/verbal_model.json",
    ) -> None:
        self.chat_controller = chat_controller
        self.trainer = trainer
        self.evaluator = evaluator
        self.teacher_generator = teacher_generator
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

    @staticmethod
    def _normalize_text(text: str) -> str:
        s = str(text or "").strip()
        return s.replace(" ", "").replace("　", "")

    @staticmethod
    def _char_jaccard(a: str, b: str) -> float:
        sa = set(ChatLearningCentral._normalize_text(a))
        sb = set(ChatLearningCentral._normalize_text(b))
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

    @staticmethod
    def _normalize_generated_pair(row: Any) -> Dict[str, Any]:
        if isinstance(row, dict):
            return {
                "input_tokens": [str(x).strip() for x in row.get("input_tokens", []) if str(x).strip()],
                "target_tokens": [str(x).strip() for x in row.get("target_tokens", []) if str(x).strip()],
                "target_text": str(row.get("target_text", "")).strip(),
            }
        if isinstance(row, list):
            return {
                "input_tokens": [str(x).strip() for x in row if str(x).strip()],
                "target_tokens": [],
                "target_text": "",
            }
        return {
            "input_tokens": [],
            "target_tokens": [],
            "target_text": "",
        }

    @staticmethod
    def _strip_punct_tokens(tokens: Sequence[str]) -> List[str]:
        punct = {"。", "、", "？", "！", "?", "!"}
        return [str(x).strip() for x in tokens if str(x).strip() and str(x).strip() not in punct]

    @staticmethod
    def _candidate_len_penalty(text: str) -> float:
        t = str(text or "").strip()
        if not t:
            return 2.0
        n = len(t)
        if n <= 8:
            return 0.0
        if n <= 14:
            return 0.25
        if n <= 22:
            return 0.75
        return 1.4

    def _teacher_priority_score(
        self,
        text: str,
        target_tokens: Sequence[str],
        target_text: str,
    ) -> Dict[str, float]:
        clean_target_tokens = self._strip_punct_tokens(target_tokens)
        token_hit = 0.0
        if clean_target_tokens:
            token_hit = self._token_f1(
                [tok for tok in clean_target_tokens if tok in text],
                clean_target_tokens,
            )

        text_sim = self._char_jaccard(text, target_text) if target_text else 0.0
        short_bonus = 0.35 if len(str(text or "").strip()) <= max(8, len(target_text) + 2) else 0.0
        len_penalty = self._candidate_len_penalty(text)

        total = (token_hit * 5.5) + (text_sim * 6.0) + short_bonus - len_penalty
        return {
            "token_hit": round(token_hit, 6),
            "text_sim": round(text_sim, 6),
            "short_bonus": round(short_bonus, 6),
            "len_penalty": round(len_penalty, 6),
            "teacher_priority_total": round(total, 6),
        }

    def _choose_teacher_forced_candidate(
        self,
        candidates: List[str],
        target_tokens: Sequence[str],
        target_text: str,
    ) -> Optional[Dict[str, Any]]:
        if not candidates:
            return None
        if not target_tokens and not target_text:
            return None

        scored: List[Dict[str, Any]] = []
        for text in candidates:
            metrics = self._teacher_priority_score(
                text=text,
                target_tokens=target_tokens,
                target_text=target_text,
            )
            scored.append(
                {
                    "text": text,
                    **metrics,
                }
            )

        scored.sort(key=lambda x: float(x["teacher_priority_total"]), reverse=True)
        best = scored[0]

        # teacher との近さが十分なら外部 evaluator を通さず即採用
        strong_match = (
            best["text_sim"] >= 0.62
            or best["token_hit"] >= 0.72
            or (
                best["text_sim"] >= 0.42
                and best["token_hit"] >= 0.45
                and len(str(best["text"]).strip()) <= max(10, len(target_text) + 2)
            )
        )
        if strong_match:
            return {
                "best_text": str(best["text"]),
                "scores": scored,
                "reason": "teacher_forced_priority",
            }

        return None

    def generate_teacher_pair(self) -> Dict[str, Any]:
        if self.teacher_generator is None:
            raise RuntimeError("teacher_generator is not set")
        return self._normalize_generated_pair(self.teacher_generator.generate())

    def run_once(
        self,
        input_tokens: Sequence[str],
        depth: int = 4,
        target_tokens: Optional[Sequence[str]] = None,
        target_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        started = time.time()
        episode_id = f"chat_ep_{uuid.uuid4().hex[:12]}"

        input_tokens = [str(x).strip() for x in input_tokens if str(x).strip()]
        if not input_tokens:
            raise ValueError("input_tokens is empty")

        target_tokens_clean = [str(x).strip() for x in (target_tokens or []) if str(x).strip()]
        target_text_clean = str(target_text).strip() if target_text is not None else ""

        self._log("=" * 88, force=True)
        self._log(f"[CHAT_CENTRAL] START id={episode_id} depth={depth}", force=True)
        self._log(
            f"[CHAT_CENTRAL] INPUT count={len(input_tokens)} tokens={self._preview_tokens(input_tokens, 40)}",
            force=True,
        )

        if target_tokens_clean or target_text_clean:
            self._log(
                f"[CHAT_CENTRAL] TARGET tokens={self._preview_tokens(target_tokens_clean, 40)} "
                f"text={json.dumps(target_text_clean, ensure_ascii=False)}",
                force=True,
            )

        if target_tokens_clean or target_text_clean:
            recursive_episode = self.trainer.run_supervised_recursive_episode(
                input_tokens=input_tokens,
                chat_controller=self.chat_controller,
                evaluator=self.evaluator,
                depth=depth,
                target_tokens=target_tokens_clean,
                target_text=target_text_clean,
            )
        else:
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
            target_tokens=target_tokens_clean,
            target_text=target_text_clean,
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
            "target_tokens": list(target_tokens_clean),
            "target_text": target_text_clean,
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
                "target_token_count": len(target_tokens_clean),
                "final_output_count": len(final_output_tokens),
                "recursive_step_count": int(recursive_episode.get("recursive_step_count", 0)),
                "best_text_target_char_jaccard": round(
                    self._char_jaccard(best_text, target_text_clean) if target_text_clean else 0.0,
                    6,
                ),
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
        target_tokens: Optional[List[str]] = None,
        target_text: str = "",
    ) -> Dict[str, Any]:
        target_tokens = [str(x).strip() for x in (target_tokens or []) if str(x).strip()]
        target_text = str(target_text or "").strip()

        if not candidates:
            return {
                "best_text": "",
                "scores": [],
                "reason": "no_candidates",
            }

        # まず teacher 強制優先
        forced = self._choose_teacher_forced_candidate(
            candidates=candidates,
            target_tokens=target_tokens,
            target_text=target_text,
        )
        if forced is not None:
            self._log(
                f"[CHAT_CENTRAL][TEACHER_FORCE] selected={json.dumps(forced['best_text'], ensure_ascii=False)}",
                force=True,
            )
            return forced

        # target があるときは external evaluator より先に local rescoring を使う
        if target_tokens or target_text:
            model = load_verbal_model(self.verbal_model_path)
            local_scores = score_candidates_locally(
                candidates=candidates,
                final_tokens=final_tokens,
                original_input=input_tokens,
                model=model,
            )

            rescored: List[Dict[str, Any]] = []
            for row in local_scores:
                text = str(row.get("text", ""))
                base_score = float(row.get("score", 0.0))
                teacher_metrics = self._teacher_priority_score(
                    text=text,
                    target_tokens=target_tokens,
                    target_text=target_text,
                )

                total = base_score + teacher_metrics["teacher_priority_total"]
                rescored.append(
                    {
                        "text": text,
                        "score": round(total, 6),
                        "base_score": round(base_score, 6),
                        **teacher_metrics,
                    }
                )

            rescored.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
            return {
                "best_text": rescored[0]["text"] if rescored else candidates[0],
                "scores": rescored,
                "reason": "local_teacher_hard_priority",
            }

        # teacher なしのときだけ external evaluator を使う
        method = getattr(self.evaluator, "evaluate_verbal_candidates", None)
        if callable(method):
            try:
                result = method(
                    input_tokens=input_tokens,
                    final_tokens=final_tokens,
                    candidates=candidates,
                    target_tokens=target_tokens,
                    target_text=target_text,
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

    teacher_generator = GeminiTeacherPairGenerator(
        model_name=getattr(args, "teacher_generator_model", None) or "gemini-2.5-flash-lite",
        enabled=bool(getattr(args, "auto_teacher", False)),
        verbose=bool(getattr(args, "verbose", False)),
    )

    return ChatLearningCentral(
        chat_controller=controller,
        trainer=trainer,
        evaluator=evaluator,
        teacher_generator=teacher_generator,
        verbose=args.verbose,
        save_path=args.save_path,
        verbal_model_path=args.verbal_model_path,
    )