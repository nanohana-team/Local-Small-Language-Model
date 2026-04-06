from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.core.io.lsd_lexicon import load_lexicon_container
from src.core.logging.trace_logger import JsonlTraceLogger
from src.core.planner.intent_planner import plan_intent
from src.core.recall.semantic_recall import recall_semantics
from src.core.scoring.basic_scorer import choose_best_response
from src.core.schema import (
    DialogueState,
    LexiconContainer,
    TraceLog,
    build_input_state,
    new_session_id,
    new_turn_id,
)
from src.core.slots.slot_filler import fill_slots
from src.core.surface.surface_realizer import realize_surface

LOGGER = logging.getLogger(__name__)
JST = ZoneInfo("Asia/Tokyo")


class SurfaceNormalizer:
    def __init__(self, lexicon: LexiconContainer) -> None:
        self.lexicon = lexicon
        self.surface_map = self._build_surface_map(lexicon)
        self.length_index = self._build_length_index(self.surface_map.keys())

    def normalize_token(self, token: str) -> List[str]:
        text = str(token).strip()
        if not text:
            return []
        if text in self.surface_map:
            return list(self.surface_map[text])
        return [text]

    def normalize_text(self, raw_text: str) -> List[str]:
        text = str(raw_text or "").strip()
        if not text:
            return []

        tokens: List[str] = []
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if ch.isspace():
                i += 1
                continue

            matched = self._longest_match(text, i)
            if matched:
                tokens.extend(self.surface_map[matched])
                i += len(matched)
                continue

            tokens.append(ch)
            i += 1
        return tokens

    def _build_surface_map(self, lexicon: LexiconContainer) -> Dict[str, List[str]]:
        mapping: Dict[str, List[str]] = {}
        for entry in lexicon.entries.values():
            if entry.word:
                mapping.setdefault(entry.word, [entry.word])
            for alias in entry.aliases:
                alias_text = str(alias).strip()
                if alias_text:
                    mapping.setdefault(alias_text, [entry.word])
            for form in entry.surface_forms:
                surface = str(form.surface).strip()
                if not surface:
                    continue
                mapping[surface] = list(form.tokens or [entry.word])
        return mapping

    def _build_length_index(self, words: Iterable[str]) -> Dict[int, set[str]]:
        index: Dict[int, set[str]] = {}
        for word in words:
            w = str(word).strip()
            if not w:
                continue
            index.setdefault(len(w), set()).add(w)
        return index

    def _longest_match(self, text: str, start: int) -> str:
        remaining = len(text) - start
        lengths = sorted(
            (length for length in self.length_index if length <= remaining),
            reverse=True,
        )
        for length in lengths:
            piece = text[start : start + length]
            if piece in self.length_index[length]:
                return piece
        return ""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LSLM v3 minimal chat runner")
    parser.add_argument("--lexicon", default="libs/dict.lsdx", help="辞書ファイルパス (.json / .lsd / .lsdx)")
    parser.add_argument("--text", default="", help="入力テキスト")
    parser.add_argument("--words", nargs="*", default=None, help="すでに分かち書き済みの入力トークン列")
    parser.add_argument("--trace-dir", default="runtime/traces", help="trace JSONL の保存先ディレクトリ")
    parser.add_argument("--no-trace", action="store_true", help="trace JSONL を保存しない")
    parser.add_argument("--console-debug", action="store_true", help="main 側互換用フラグ")
    return parser.parse_args(argv)


def load_lexicon(path: Path) -> LexiconContainer:
    LOGGER.info("lexicon.load.start path=%s", path)
    container_raw = load_lexicon_container(path)
    lexicon = LexiconContainer.from_dict(container_raw)
    LOGGER.info("lexicon.load.done entries=%s version=%s", len(lexicon.entries), lexicon.meta.version)
    return lexicon


def build_raw_text(args: argparse.Namespace) -> str:
    if args.words:
        return " ".join(str(word) for word in args.words if str(word).strip())
    return str(args.text or "").strip()


def build_tokens(
    raw_text: str,
    explicit_words: Optional[Sequence[str]],
    normalizer: SurfaceNormalizer,
) -> List[str]:
    if explicit_words:
        tokens: List[str] = []
        for word in explicit_words:
            tokens.extend(normalizer.normalize_token(str(word)))
        return [token for token in tokens if token]

    if not raw_text:
        return []

    if " " in raw_text:
        tokens: List[str] = []
        for part in raw_text.split():
            tokens.extend(normalizer.normalize_token(part))
        return [token for token in tokens if token]

    return normalizer.normalize_text(raw_text)


def run_pipeline(
    args: argparse.Namespace,
    lexicon: LexiconContainer,
    normalizer: SurfaceNormalizer,
) -> Tuple[str, Optional[Path]]:
    lexicon_path = Path(args.lexicon)
    if not lexicon_path.exists():
        LOGGER.error("lexicon_not_found path=%s", lexicon_path)
        raise FileNotFoundError(f"Lexicon file not found: {lexicon_path}")

    LOGGER.info("minimal_chat.start lexicon=%s", lexicon_path)

    raw_text = build_raw_text(args)
    if not raw_text.strip():
        LOGGER.error("empty_input")
        raise ValueError("Input text is empty. Use --text or --words.")

    tokens = build_tokens(raw_text=raw_text, explicit_words=args.words, normalizer=normalizer)
    LOGGER.info("input raw_text=%s", raw_text)
    LOGGER.info("input tokens=%s", tokens)

    session_id = new_session_id()
    turn_id = new_turn_id()

    input_state = build_input_state(
        raw_text=raw_text,
        tokens=tokens,
        normalized_tokens=tokens,
        session_id=session_id,
        turn_id=turn_id,
        timestamp=datetime.now(JST).isoformat(timespec="seconds"),
    )
    dialogue_state = DialogueState()

    intent_plan = plan_intent(input_state=input_state, dialogue_state=dialogue_state)
    recall_result = recall_semantics(
        input_state=input_state,
        lexicon=lexicon,
        dialogue_state=dialogue_state,
        intent_plan=intent_plan,
    )
    filled_slots = fill_slots(
        input_state=input_state,
        recall_result=recall_result,
        lexicon=lexicon,
        intent_plan=intent_plan,
        dialogue_state=dialogue_state,
    )
    surface_plan, candidates = realize_surface(
        filled_slots=filled_slots,
        intent_plan=intent_plan,
        lexicon=lexicon,
    )
    response, scored_candidates = choose_best_response(
        input_state=input_state,
        intent_plan=intent_plan,
        filled_slots=filled_slots,
        candidates=candidates,
    )

    trace = TraceLog(
        session_id=session_id,
        turn_id=turn_id,
        input_state=input_state,
        dialogue_state=dialogue_state,
        intent_plan=intent_plan,
        recall_result=recall_result,
        filled_slots=filled_slots,
        surface_plan=surface_plan,
        candidates=scored_candidates,
        response=response,
        evaluation=[],
        debug={
            "lexicon_path": str(lexicon_path),
            "entry_count": len(lexicon.entries),
            "raw_text": raw_text,
            "tokens": tokens,
        },
    )

    trace_path: Optional[Path] = None
    if not args.no_trace:
        trace_logger = JsonlTraceLogger(
            log_dir=args.trace_dir,
            latest_name="latest_trace.jsonl",
            rotate_on_start=False,
        )
        trace_path = trace_logger.append_trace(trace)
        LOGGER.info("trace_saved path=%s", trace_path)

    LOGGER.info("response chosen=%s total=%.4f", response.text, response.score.total)
    return response.text, trace_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    # --- logging（単体実行用） ---
    from src.utils.logging import setup_logging
    import logging

    setup_logging(
        app_name="lslm_minimal_chat",
        console_level=logging.DEBUG if args.console_debug else logging.INFO,
    )

    # --- ここでだけ辞書ロード ---
    lexicon_path = Path(args.lexicon)
    if not lexicon_path.exists():
        raise FileNotFoundError(f"Lexicon file not found: {lexicon_path}")

    lexicon = load_lexicon(lexicon_path)
    normalizer = SurfaceNormalizer(lexicon)

    # --- 実行 ---
    response_text, _ = run_pipeline(
        args,
        lexicon,
        normalizer,
    )

    sys.stdout.write(response_text + "\n")


if __name__ == "__main__":
    main()