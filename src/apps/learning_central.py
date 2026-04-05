from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional

from src.core.primitive.divergence import DivergenceModel
from src.core.primitive.convergence import ConvergenceModel
from src.llm.evaluator_gemini import GeminiEvaluator
from src.llm.input_generator_gemini import GeminiInputGenerator
from src.utils.storage import StorageManager
from src.utils.trainer import Trainer


class LearningCentral:
    def __init__(
        self,
        divergence_model: DivergenceModel,
        convergence_model: ConvergenceModel,
        evaluator: GeminiEvaluator,
        storage: StorageManager,
        trainer: Trainer,
        verbose: bool = False,
        input_generator: GeminiInputGenerator | None = None,
    ) -> None:
        self.divergence_model = divergence_model
        self.convergence_model = convergence_model
        self.evaluator = evaluator
        self.storage = storage
        self.trainer = trainer
        self.verbose = verbose
        self.input_generator = input_generator

    def _log(self, message: str, force: bool = False) -> None:
        if force or self.verbose:
            print(message, flush=True)

    @staticmethod
    def _preview_tokens(tokens: List[str], limit: int = 20) -> str:
        if not tokens:
            return "[]"
        shown = tokens[:limit]
        suffix = "" if len(tokens) <= limit else f" ... (+{len(tokens) - limit})"
        return json.dumps(shown, ensure_ascii=False) + suffix

    @staticmethod
    def _safe_score_from_episode(episode: Dict[str, Any]) -> float | None:
        evaluation = episode.get("evaluation") or {}
        try:
            value = evaluation.get("score_total")
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    def build_response_text(self, final_tokens: List[str]) -> str:
        if not final_tokens:
            return "……"

        text = "".join(final_tokens)

        replacements = [
            ("。。", "。"),
            ("、、", "、"),
            ("，，", "，"),
            ("！！", "！"),
            ("？？", "？"),
            ("  ", " "),
        ]
        for src, dst in replacements:
            while src in text:
                text = text.replace(src, dst)

        if text and text[-1] not in "。！？…":
            text += "。"

        return text

    def run_episode(
        self,
        input_tokens: List[str],
        depth: int,
        evaluate: bool = True,
    ) -> Dict[str, Any]:
        started_at = time.time()
        episode_id = f"ep_{uuid.uuid4().hex[:12]}"

        episode: Dict[str, Any] = {
            "episode_id": episode_id,
            "input_tokens": list(input_tokens),
            "depth": int(depth),
            "timestamps": {
                "started_at_unix": started_at,
            },
            "initial_core": [],
            "divergence_steps": [],
            "mid_converged": [],
            "final_expanded": [],
            "response_text": "",
            "evaluation": None,
            "metrics": {},
        }

        self._log("=" * 88, force=True)
        self._log(
            f"[EPISODE] START id={episode_id} depth={depth} evaluate={evaluate}",
            force=True,
        )
        self._log(
            f"[EPISODE] INPUT count={len(input_tokens)} tokens={self._preview_tokens(input_tokens, 40)}",
            force=True,
        )

        initial_core = self.convergence_model.initial_converge(
            input_tokens=input_tokens,
            target_min=3,
            target_max=20,
        )
        episode["initial_core"] = list(initial_core)
        self._log(
            f"[STEP][INITIAL_CONVERGE] count={len(initial_core)} tokens={self._preview_tokens(initial_core, 40)}",
            force=True,
        )

        multi = self.divergence_model.multi_expand(
            tokens=initial_core,
            depth=depth,
            top_k=self.divergence_model.default_branch,
            allow_function_words=True,
        )
        episode["divergence_steps"] = list(multi["steps"])
        current_tokens = list(multi["output_tokens"])

        self._log(
            f"[STEP][DIVERGENCE] depth_steps={len(multi['steps'])} output_count={len(current_tokens)} "
            f"output={self._preview_tokens(current_tokens, 40)}",
            force=True,
        )
        for step in multi["steps"]:
            expanded_nodes = step.get("expanded", [])
            output_tokens = step.get("output_tokens", [])
            self._log(
                f"[STEP][DIVERGENCE][LAYER {step.get('depth')}] "
                f"in={len(step.get('input_tokens', []))} sources={len(expanded_nodes)} out={len(output_tokens)}",
                force=True,
            )
            if self.verbose:
                for node in expanded_nodes[:12]:
                    self._log(
                        f"  [EXPAND] src={node.get('token')} -> children={self._preview_tokens(node.get('children', []), 20)}",
                        force=False,
                    )
                if len(expanded_nodes) > 12:
                    self._log(f"  [EXPAND] ... (+{len(expanded_nodes) - 12} more sources)", force=False)

        mid_converged = self.convergence_model.converge(
            candidate_tokens=current_tokens,
            original_input=input_tokens,
            initial_core=initial_core,
        )
        episode["mid_converged"] = list(mid_converged)
        self._log(
            f"[STEP][MID_CONVERGE] count={len(mid_converged)} tokens={self._preview_tokens(mid_converged, 40)}",
            force=True,
        )

        final_expanded = self.divergence_model.final_expand(
            tokens=mid_converged,
            original_input=input_tokens,
        )
        episode["final_expanded"] = list(final_expanded)
        self._log(
            f"[STEP][FINAL_EXPAND] count={len(final_expanded)} tokens={self._preview_tokens(final_expanded, 40)}",
            force=True,
        )

        response_text = self.build_response_text(final_expanded)
        episode["response_text"] = response_text
        self._log(
            f"[STEP][RESPONSE] chars={len(response_text)} text={json.dumps(response_text, ensure_ascii=False)}",
            force=True,
        )

        if evaluate:
            self._log("[STEP][EVALUATION] start", force=True)
            evaluation = self.evaluator.evaluate_episode(episode)
            episode["evaluation"] = evaluation
            self._log(
                f"[STEP][EVALUATION] done payload={json.dumps(evaluation, ensure_ascii=False)}",
                force=True,
            )

        ended_at = time.time()
        episode["timestamps"]["ended_at_unix"] = ended_at
        episode["metrics"] = {
            "elapsed_sec": round(ended_at - started_at, 6),
            "input_token_count": len(input_tokens),
            "initial_core_count": len(initial_core),
            "mid_converged_count": len(mid_converged),
            "final_expanded_count": len(final_expanded),
            "response_char_count": len(response_text),
        }

        self._log(
            f"[EPISODE] END id={episode_id} elapsed={episode['metrics']['elapsed_sec']:.3f}s "
            f"metrics={json.dumps(episode['metrics'], ensure_ascii=False)}",
            force=True,
        )
        return episode

    def run_chat_once(
        self,
        input_tokens: List[str],
        depth: int,
    ) -> Dict[str, Any]:
        episode = self.run_episode(
            input_tokens=input_tokens,
            depth=depth,
            evaluate=False,
        )
        return {
            "input_tokens": episode["input_tokens"],
            "initial_core": episode["initial_core"],
            "mid_converged": episode["mid_converged"],
            "final_expanded": episode["final_expanded"],
            "response_text": episode["response_text"],
            "metrics": episode["metrics"],
        }

    def _log_buffer_snapshot(self, buffer: List[Dict[str, Any]]) -> None:
        scores = [self._safe_score_from_episode(ep) for ep in buffer]
        valid_scores = [s for s in scores if s is not None]
        if not valid_scores:
            self._log(f"[BUFFER] size={len(buffer)} score_stats=none", force=True)
            return

        avg_score = sum(valid_scores) / len(valid_scores)
        self._log(
            f"[BUFFER] size={len(buffer)} "
            f"score_avg={avg_score:.3f} score_min={min(valid_scores):.3f} score_max={max(valid_scores):.3f}",
            force=True,
        )

    def _run_train_update(self, buffer: List[Dict[str, Any]]) -> None:
        if not buffer:
            self._log("[TRAIN] skip update: empty buffer", force=True)
            return

        self._log(f"[TRAIN] update start buffer_size={len(buffer)}", force=True)
        self._log_buffer_snapshot(buffer)

        div_before = dict(self.divergence_model.state.get("weights", {}))
        conv_keep_before = dict(self.convergence_model.state.get("token_keep_bias", {}))
        conv_drop_before = dict(self.convergence_model.state.get("token_drop_bias", {}))

        self.trainer.update_models(
            episodes=buffer,
            divergence_model=self.divergence_model,
            convergence_model=self.convergence_model,
        )

        div_after = dict(self.divergence_model.state.get("weights", {}))
        conv_keep_after = dict(self.convergence_model.state.get("token_keep_bias", {}))
        conv_drop_after = dict(self.convergence_model.state.get("token_drop_bias", {}))

        if div_before or div_after:
            axis_changes: List[tuple[str, float, float, float]] = []
            for axis in sorted(set(div_before) | set(div_after)):
                before = float(div_before.get(axis, 1.0))
                after = float(div_after.get(axis, 1.0))
                delta = after - before
                axis_changes.append((axis, before, after, delta))
            axis_changes.sort(key=lambda x: abs(x[3]), reverse=True)

            self._log("[TRAIN][DIVERGENCE] top axis changes:", force=True)
            for axis, before, after, delta in axis_changes[:10]:
                self._log(
                    f"  [AXIS] {axis}: {before:.6f} -> {after:.6f} (delta={delta:+.6f})",
                    force=True,
                )

        keep_changes: List[tuple[str, float, float, float]] = []
        for token in set(conv_keep_before) | set(conv_keep_after):
            before = float(conv_keep_before.get(token, 0.0))
            after = float(conv_keep_after.get(token, 0.0))
            delta = after - before
            if abs(delta) > 1e-12:
                keep_changes.append((token, before, after, delta))
        keep_changes.sort(key=lambda x: abs(x[3]), reverse=True)

        drop_changes: List[tuple[str, float, float, float]] = []
        for token in set(conv_drop_before) | set(conv_drop_after):
            before = float(conv_drop_before.get(token, 0.0))
            after = float(conv_drop_after.get(token, 0.0))
            delta = after - before
            if abs(delta) > 1e-12:
                drop_changes.append((token, before, after, delta))
        drop_changes.sort(key=lambda x: abs(x[3]), reverse=True)

        self._log("[TRAIN][CONVERGENCE] top keep_bias changes:", force=True)
        for token, before, after, delta in keep_changes[:15]:
            self._log(
                f"  [KEEP] {token}: {before:.6f} -> {after:.6f} (delta={delta:+.6f})",
                force=True,
            )
        if not keep_changes:
            self._log("  [KEEP] no changes", force=True)

        self._log("[TRAIN][CONVERGENCE] top drop_bias changes:", force=True)
        for token, before, after, delta in drop_changes[:15]:
            self._log(
                f"  [DROP] {token}: {before:.6f} -> {after:.6f} (delta={delta:+.6f})",
                force=True,
            )
        if not drop_changes:
            self._log("  [DROP] no changes", force=True)

        self._log("[TRAIN] update done", force=True)

    def run_learning_loop(
        self,
        dataset: Optional[List[List[str]]],
        depth: int,
        update_interval: int,
        max_episodes: Optional[int] = None,
        auto_input: bool = False,
    ) -> None:
        buffer: List[Dict[str, Any]] = []

        self._log(
            f"[LEARNING] loop start auto_input={auto_input} depth={depth} "
            f"update_interval={update_interval} max_episodes={max_episodes}",
            force=True,
        )

        if auto_input:
            if max_episodes is None:
                max_episodes = 100

            for idx in range(1, max_episodes + 1):
                if self.input_generator is None:
                    raise RuntimeError("auto_input=True but input_generator is not set")

                input_tokens = self.input_generator.generate()
                self._log(
                    f"[LEARNING] episode {idx}/{max_episodes} generated_input="
                    f"{self._preview_tokens(input_tokens, 40)}",
                    force=True,
                )

                episode = self.run_episode(
                    input_tokens=input_tokens,
                    depth=depth,
                    evaluate=True,
                )
                self.storage.save_episode(episode)
                buffer.append(episode)

                score = self._safe_score_from_episode(episode)
                self._log(
                    f"[LEARNING] saved episode_id={episode['episode_id']} "
                    f"score={score} elapsed={episode['metrics']['elapsed_sec']:.3f}s "
                    f"buffer_size={len(buffer)}/{update_interval}",
                    force=True,
                )

                if len(buffer) >= update_interval:
                    self._run_train_update(buffer)
                    buffer.clear()

            if buffer:
                self._log(f"[TRAIN] final update start buffer_size={len(buffer)}", force=True)
                self._run_train_update(buffer)
                buffer.clear()
                self._log("[TRAIN] final update done", force=True)
            return

        dataset = dataset or []
        total = min(len(dataset), max_episodes) if max_episodes is not None else len(dataset)

        for idx, input_tokens in enumerate(dataset[:total], start=1):
            self._log(
                f"[LEARNING] episode {idx}/{total} input={self._preview_tokens(input_tokens, 40)}",
                force=True,
            )

            episode = self.run_episode(
                input_tokens=input_tokens,
                depth=depth,
                evaluate=True,
            )
            self.storage.save_episode(episode)
            buffer.append(episode)

            score = self._safe_score_from_episode(episode)
            self._log(
                f"[LEARNING] saved episode_id={episode['episode_id']} "
                f"score={score} elapsed={episode['metrics']['elapsed_sec']:.3f}s "
                f"buffer_size={len(buffer)}/{update_interval}",
                force=True,
            )

            if len(buffer) >= update_interval:
                self._run_train_update(buffer)
                buffer.clear()

        if buffer:
            self._log(f"[TRAIN] final update start buffer_size={len(buffer)}", force=True)
            self._run_train_update(buffer)
            buffer.clear()
            self._log("[TRAIN] final update done", force=True)