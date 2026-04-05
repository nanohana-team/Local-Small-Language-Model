from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.utils.verbal import verbalize


@dataclass
class ChatResult:
    input_tokens: List[str]
    thought_core: List[str]
    thought_expanded: List[str]
    thought_converged: List[str]
    final_expanded: List[str]
    final_converged: List[str]
    response_text: str
    mode: str
    depth: int
    elapsed_ms: float
    trace: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "thought_core": self.thought_core,
            "thought_expanded": self.thought_expanded,
            "thought_converged": self.thought_converged,
            "final_expanded": self.final_expanded,
            "final_converged": self.final_converged,
            "response_text": self.response_text,
            "mode": self.mode,
            "depth": self.depth,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "trace": self.trace,
        }


class ChatController:
    """
    chat mode execution controller for LSLM

    flow:
      1) normalize input
      2) build thought core        (initial converge)
      3) expand thought           (multi_expand)
      4) converge thought         (converge)
      5) final expand for wording (depth=3, branch=4)
      6) final converge to ~10 tokens
      7) verbalize response text
    """

    def __init__(
        self,
        divergence_model: Any,
        convergence_model: Any,
        evaluator: Optional[Any] = None,
        storage: Optional[Any] = None,
        trainer: Optional[Any] = None,
        verbose: bool = False,
    ) -> None:
        self.divergence_model = divergence_model
        self.convergence_model = convergence_model
        self.evaluator = evaluator
        self.storage = storage
        self.trainer = trainer
        self.verbose = verbose

    # =========================================================
    # public
    # =========================================================

    def think_once(self, input_tokens: Sequence[str], depth: int = 4) -> Dict[str, Any]:
        tokens = self._normalize_tokens(input_tokens)
        mode = self._decide_mode(tokens=tokens, requested_depth=depth)
        effective_depth = self._resolve_depth(mode=mode, requested_depth=depth)

        self._log("=" * 88, force=True)
        self._log(
            f"[CHAT] START input_count={len(tokens)} depth={effective_depth} mode={mode}",
            force=True,
        )
        self._log(
            f"[CHAT] INPUT tokens={self._preview_tokens(tokens, 40)}",
            force=True,
        )

        thought_core = self._initial_converge(tokens)
        self._log(
            f"[CHAT][THOUGHT_CORE] count={len(thought_core)} tokens={self._preview_tokens(thought_core, 40)}",
            force=True,
        )

        thought_expand_result = self._multi_expand(
            tokens=thought_core,
            depth=effective_depth,
            top_k=self._get_default_branch(),
            allow_function_words=True,
        )
        thought_steps = thought_expand_result.get("steps", [])
        thought_expanded = list(thought_expand_result.get("output_tokens", []))

        self._log(
            f"[CHAT][THOUGHT_EXPAND] depth_steps={len(thought_steps)} output_count={len(thought_expanded)} "
            f"tokens={self._preview_tokens(thought_expanded, 40)}",
            force=True,
        )

        thought_converged = self._converge(
            candidate_tokens=thought_expanded,
            original_input=tokens,
            initial_core=thought_core,
            target_min=4,
            target_max=16,
        )
        self._log(
            f"[CHAT][THOUGHT_CONVERGE] count={len(thought_converged)} tokens={self._preview_tokens(thought_converged, 40)}",
            force=True,
        )

        final_expand_result = self._multi_expand(
            tokens=thought_converged,
            depth=3,
            top_k=4,
            allow_function_words=True,
        )
        final_expand_steps = final_expand_result.get("steps", [])
        final_expanded = list(final_expand_result.get("output_tokens", []))

        self._log(
            f"[CHAT][FINAL_EXPAND] depth_steps={len(final_expand_steps)} output_count={len(final_expanded)} "
            f"tokens={self._preview_tokens(final_expanded, 40)}",
            force=True,
        )

        final_converged = self._final_converge_to_response_tokens(
            candidate_tokens=final_expanded,
            original_input=tokens,
            thought_tokens=thought_converged,
            target_min=8,
            target_max=10,
        )
        self._log(
            f"[CHAT][FINAL_CONVERGE] count={len(final_converged)} tokens={self._preview_tokens(final_converged, 40)}",
            force=True,
        )

        return {
            "input_tokens": list(tokens),
            "thought_core": list(thought_core),
            "thought_expanded": list(thought_expanded),
            "thought_converged": list(thought_converged),
            "final_expanded": list(final_expanded),
            "final_converged": list(final_converged),
            "mode": mode,
            "depth": effective_depth,
            "trace": {
                "thought_expand_steps": thought_steps,
                "final_expand_steps": final_expand_steps,
            },
        }

    def run_once(self, input_tokens: Sequence[str], depth: int = 4) -> Dict[str, Any]:
        started = time.perf_counter()
        thought = self.think_once(input_tokens=input_tokens, depth=depth)

        response_text = verbalize(
            final_tokens=thought["final_converged"],
            original_input=thought["input_tokens"],
        )
        self._log(
            f"[CHAT][RESPONSE] chars={len(response_text)} text={json.dumps(response_text, ensure_ascii=False)}",
            force=True,
        )

        elapsed_ms = (time.perf_counter() - started) * 1000.0

        result = ChatResult(
            input_tokens=list(thought["input_tokens"]),
            thought_core=list(thought["thought_core"]),
            thought_expanded=list(thought["thought_expanded"]),
            thought_converged=list(thought["thought_converged"]),
            final_expanded=list(thought["final_expanded"]),
            final_converged=list(thought["final_converged"]),
            response_text=response_text,
            mode=thought["mode"],
            depth=thought["depth"],
            elapsed_ms=elapsed_ms,
            trace=dict(thought["trace"]),
        )

        self._log(
            f"[CHAT] END elapsed_ms={elapsed_ms:.3f}",
            force=True,
        )

        return result.to_dict()

    # =========================================================
    # helpers
    # =========================================================

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

    def _normalize_tokens(self, tokens: Sequence[str]) -> List[str]:
        result: List[str] = []
        for token in tokens:
            t = str(token).strip()
            if t:
                result.append(t)
        return result

    def _decide_mode(self, tokens: Sequence[str], requested_depth: int) -> str:
        if requested_depth >= 6:
            return "reasoning"
        if len(tokens) <= 4:
            return "fast"
        return "normal"

    def _resolve_depth(self, mode: str, requested_depth: int) -> int:
        if mode == "fast":
            return max(1, min(requested_depth, 2))
        if mode == "normal":
            return max(2, min(requested_depth, 4))
        return max(3, requested_depth)

    def _get_default_branch(self) -> int:
        value = getattr(self.divergence_model, "default_branch", None)
        if isinstance(value, int) and value > 0:
            return value
        return 6

    # =========================================================
    # core steps
    # =========================================================

    def _initial_converge(self, input_tokens: Sequence[str]) -> List[str]:
        method = getattr(self.convergence_model, "initial_converge", None)
        if callable(method):
            try:
                output = method(
                    input_tokens=list(input_tokens),
                    target_min=3,
                    target_max=20,
                )
                tokens = self._coerce_tokens(output)
                if tokens:
                    return tokens
            except Exception as exc:
                self._log(f"[WARN][INITIAL_CONVERGE] fallback reason={exc}", force=True)

        return self._dedupe_keep_order(list(input_tokens))[:20]

    def _multi_expand(
        self,
        tokens: Sequence[str],
        depth: int,
        top_k: int,
        allow_function_words: bool,
    ) -> Dict[str, Any]:
        method = getattr(self.divergence_model, "multi_expand", None)
        if callable(method):
            try:
                output = method(
                    tokens=list(tokens),
                    depth=depth,
                    top_k=top_k,
                    allow_function_words=allow_function_words,
                )
                if isinstance(output, dict):
                    return {
                        "steps": list(output.get("steps", [])),
                        "output_tokens": self._coerce_tokens(output.get("output_tokens", [])),
                    }
            except Exception as exc:
                self._log(f"[WARN][MULTI_EXPAND] fallback reason={exc}", force=True)

        expanded = self._dedupe_keep_order(list(tokens))
        return {
            "steps": [
                {
                    "depth": 1,
                    "input_tokens": list(tokens),
                    "expanded": [],
                    "output_tokens": expanded,
                }
            ],
            "output_tokens": expanded,
        }

    def _converge(
        self,
        candidate_tokens: Sequence[str],
        original_input: Sequence[str],
        initial_core: Sequence[str],
        target_min: int,
        target_max: int,
    ) -> List[str]:
        method = getattr(self.convergence_model, "converge", None)
        if callable(method):
            try:
                output = method(
                    candidate_tokens=list(candidate_tokens),
                    original_input=list(original_input),
                    initial_core=list(initial_core),
                )
                tokens = self._coerce_tokens(output)
                if tokens:
                    return self._trim_with_priority(
                        tokens=tokens,
                        priority_tokens=list(original_input) + list(initial_core),
                        target_min=target_min,
                        target_max=target_max,
                    )
            except Exception as exc:
                self._log(f"[WARN][CONVERGE] fallback reason={exc}", force=True)

        merged = self._dedupe_keep_order(list(original_input) + list(initial_core) + list(candidate_tokens))
        return merged[:target_max]

    def _final_converge_to_response_tokens(
        self,
        candidate_tokens: Sequence[str],
        original_input: Sequence[str],
        thought_tokens: Sequence[str],
        target_min: int = 8,
        target_max: int = 10,
    ) -> List[str]:
        method = getattr(self.convergence_model, "converge", None)

        selected: List[str] = []
        for t in original_input:
            if t not in {"。", "、"} and t not in selected:
                selected.append(t)

        if callable(method):
            try:
                output = method(
                    candidate_tokens=list(candidate_tokens),
                    original_input=list(original_input),
                    initial_core=list(thought_tokens),
                )
                tokens = self._coerce_tokens(output)
                if tokens:
                    for t in tokens:
                        if t not in selected:
                            selected.append(t)
                        if len(selected) >= target_max:
                            break
                    return selected[:target_max]
            except Exception as exc:
                self._log(f"[WARN][FINAL_CONVERGE] fallback reason={exc}", force=True)

        for t in list(thought_tokens) + list(candidate_tokens):
            if t not in selected:
                selected.append(t)
            if len(selected) >= target_max:
                break

        if len(selected) < target_min:
            return selected

        return selected[:target_max]

    # =========================================================
    # utils
    # =========================================================

    def _trim_with_priority(
        self,
        tokens: Sequence[str],
        priority_tokens: Sequence[str],
        target_min: int,
        target_max: int,
    ) -> List[str]:
        deduped = self._dedupe_keep_order(tokens)
        priority = self._dedupe_keep_order(priority_tokens)

        selected: List[str] = []

        for token in priority:
            if token in deduped and token not in selected:
                selected.append(token)
            if len(selected) >= target_max:
                return selected[:target_max]

        for token in deduped:
            if token not in selected:
                selected.append(token)
            if len(selected) >= target_max:
                return selected[:target_max]

        if len(selected) < target_min:
            return selected

        return selected[:target_max]

    def _coerce_tokens(self, output: Any) -> List[str]:
        if output is None:
            return []

        if isinstance(output, list):
            result: List[str] = []
            for item in output:
                if isinstance(item, str):
                    token = item.strip()
                    if token:
                        result.append(token)
                elif isinstance(item, dict):
                    token = item.get("token")
                    if isinstance(token, str) and token.strip():
                        result.append(token.strip())
            return result

        if isinstance(output, dict):
            for key in ("tokens", "words", "candidates", "output", "output_tokens"):
                value = output.get(key)
                if isinstance(value, list):
                    return self._coerce_tokens(value)

        if isinstance(output, str):
            return [x for x in output.strip().split() if x.strip()]

        return []

    def _dedupe_keep_order(self, tokens: Sequence[str]) -> List[str]:
        seen = set()
        result: List[str] = []
        for token in tokens:
            if token not in seen:
                seen.add(token)
                result.append(token)
        return result