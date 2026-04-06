from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

from src.core.io.lsd_lexicon import load_lexicon_container
from src.core.planner.intent_planner import plan_intent
from src.core.recall.semantic_recall import recall_semantics
from src.core.scoring.basic_scorer import choose_best_response
from src.core.schema import (
    DialogueState,
    LexiconContainer,
    TraceLog,
    build_input_state,
    dataclass_to_dict,
    new_session_id,
    new_turn_id,
)
from src.core.slots.slot_filler import fill_slots
from src.core.surface.surface_realizer import realize_surface

try:
    from src.utils.logging import setup_logging
except Exception:  # pragma: no cover
    def setup_logging(
        app_name: str = "lslm",
        console_level: int = logging.INFO,
        **_: object,
    ) -> None:
        logging.basicConfig(
            level=console_level,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LSLM v3 minimal chat runner",
    )
    parser.add_argument(
        "--lexicon",
        default="libs/dict.lsdx",
        help="辞書ファイルパス (.json / .lsd / .lsdx)",
    )
    parser.add_argument(
        "--text",
        default="",
        help="入力テキスト",
    )
    parser.add_argument(
        "--words",
        nargs="*",
        default=None,
        help="すでに分かち書き済みの入力トークン列",
    )
    parser.add_argument(
        "--trace-dir",
        default="runtime/traces",
        help="trace JSONL の保存先ディレクトリ",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="trace JSONL を保存しない",
    )
    parser.add_argument(
        "--console-debug",
        action="store_true",
        help="コンソールにも DEBUG を出す",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    setup_logging(
        app_name="lslm_minimal_chat",
        console_level=logging.DEBUG if args.console_debug else logging.INFO,
    )

    LOGGER.debug("minimal_chat.args=%s", vars(args))

    lexicon_path = Path(args.lexicon)
    LOGGER.debug("minimal_chat.lexicon_path=%s exists=%s", lexicon_path, lexicon_path.exists())

    if not lexicon_path.exists():
        LOGGER.error("lexicon_not_found path=%s", lexicon_path)
        raise FileNotFoundError(f"Lexicon file not found: {lexicon_path}")

    LOGGER.info("minimal_chat.start lexicon=%s", lexicon_path)

    lexicon = load_lexicon(lexicon_path)
    raw_text = build_raw_text(args)
    LOGGER.debug("minimal_chat.raw_text=%s", raw_text)

    if not raw_text.strip():
        LOGGER.error("empty_input")
        raise ValueError("Input text is empty. Use --text or --words.")

    tokens = build_tokens(
        raw_text=raw_text,
        explicit_words=args.words,
        lexicon=lexicon,
    )
    LOGGER.debug("minimal_chat.tokens=%s", tokens)

    session_id = new_session_id()
    turn_id = new_turn_id()
    LOGGER.debug("minimal_chat.session turn session_id=%s turn_id=%s", session_id, turn_id)

    input_state = build_input_state(
        raw_text=raw_text,
        tokens=tokens,
        normalized_tokens=tokens,
        session_id=session_id,
        turn_id=turn_id,
        timestamp=datetime.now().isoformat(timespec="seconds"),
    )
    LOGGER.debug("minimal_chat.input_state=%s", input_state)

    dialogue_state = DialogueState()
    LOGGER.debug("minimal_chat.initial_dialogue_state=%s", dialogue_state)

    LOGGER.info("input raw_text=%s", input_state.raw_text)
    LOGGER.debug("input tokens=%s", input_state.tokens)

    intent_plan = plan_intent(
        input_state=input_state,
        dialogue_state=dialogue_state,
    )
    LOGGER.info(
        "intent intent=%s confidence=%.4f policy=%s",
        intent_plan.intent,
        intent_plan.confidence,
        intent_plan.response_policy_hint,
    )
    LOGGER.debug("minimal_chat.intent_plan=%s", intent_plan)

    recall_result = recall_semantics(
        input_state=input_state,
        lexicon=lexicon,
        dialogue_state=dialogue_state,
        intent_plan=intent_plan,
    )
    LOGGER.info(
        "recall seeds=%s candidates=%s",
        recall_result.seeds,
        [c.word for c in recall_result.candidates[:10]],
    )
    LOGGER.debug(
        "minimal_chat.recall_result=%s",
        {
            "seeds": recall_result.seeds,
            "candidates": [
                {
                    "word": c.word,
                    "score": c.score,
                    "source": c.source,
                    "relation_path": c.relation_path,
                    "axis_distance": c.axis_distance,
                    "grammar_ok": c.grammar_ok,
                    "note": c.note,
                }
                for c in recall_result.candidates
            ],
        },
    )

    filled_slots = fill_slots(
        input_state=input_state,
        recall_result=recall_result,
        lexicon=lexicon,
        intent_plan=intent_plan,
        dialogue_state=dialogue_state,
    )
    LOGGER.info(
        "slots predicate=%s values=%s missing_required=%s",
        filled_slots.frame.predicate,
        {k: v.value for k, v in filled_slots.values.items()},
        filled_slots.missing_required,
    )
    LOGGER.debug("minimal_chat.filled_slots=%s", filled_slots)

    surface_plan, candidates = realize_surface(
        filled_slots=filled_slots,
        intent_plan=intent_plan,
    )
    LOGGER.info(
        "surface template_id=%s candidate_count=%s",
        surface_plan.template_id,
        len(candidates),
    )
    LOGGER.debug("minimal_chat.surface_plan=%s", surface_plan)
    LOGGER.debug("minimal_chat.surface_candidates=%s", candidates)

    response, scored_candidates = choose_best_response(
        input_state=input_state,
        intent_plan=intent_plan,
        filled_slots=filled_slots,
        candidates=candidates,
    )
    LOGGER.info(
        "response chosen=%s total=%.4f",
        response.text,
        response.score.total,
    )
    LOGGER.debug("minimal_chat.response=%s", response)
    LOGGER.debug("minimal_chat.scored_candidates=%s", scored_candidates)

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
    LOGGER.debug("minimal_chat.trace=%s", trace)

    if not args.no_trace:
        trace_path = append_trace_jsonl(
            trace=trace,
            trace_dir=Path(args.trace_dir),
        )
        LOGGER.info("trace_saved path=%s", trace_path)
    else:
        LOGGER.debug("minimal_chat.trace_skip reason=--no-trace")

    LOGGER.debug("minimal_chat.stdout_write text=%s", response.text)
    sys.stdout.write(response.text + "\n")


def load_lexicon(path: Path) -> LexiconContainer:
    LOGGER.info("lexicon.load.start path=%s", path)
    container_raw = load_lexicon_container(path)
    LOGGER.debug(
        "lexicon.load.raw_container_keys=%s",
        list(container_raw.keys()) if isinstance(container_raw, dict) else type(container_raw),
    )
    lexicon = LexiconContainer.from_dict(container_raw)
    LOGGER.info(
        "lexicon.load.done entries=%s version=%s",
        len(lexicon.entries),
        lexicon.meta.version,
    )
    LOGGER.debug(
        "lexicon.load.meta=%s indexes_summary=%s sample_words=%s",
        lexicon.meta,
        {
            "by_pos_count": len(lexicon.indexes.by_pos),
            "can_start_count": len(lexicon.indexes.can_start),
            "can_end_count": len(lexicon.indexes.can_end),
            "content_words_count": len(lexicon.indexes.content_words),
            "function_words_count": len(lexicon.indexes.function_words),
            "entry_path_count": len(lexicon.indexes.entry_path),
        },
        list(lexicon.entries.keys())[:50],
    )
    return lexicon


def build_raw_text(args: argparse.Namespace) -> str:
    if args.words:
        raw_text = " ".join(str(word) for word in args.words if str(word).strip())
        LOGGER.debug(
            "minimal_chat.build_raw_text source=words words=%s result=%s",
            args.words,
            raw_text,
        )
        return raw_text

    raw_text = str(args.text or "").strip()
    LOGGER.debug(
        "minimal_chat.build_raw_text source=text text=%s result=%s",
        args.text,
        raw_text,
    )
    return raw_text


def build_tokens(
    raw_text: str,
    explicit_words: Optional[Sequence[str]],
    lexicon: LexiconContainer,
) -> List[str]:
    if explicit_words:
        tokens = [str(word).strip() for word in explicit_words if str(word).strip()]
        LOGGER.debug(
            "minimal_chat.build_tokens source=explicit_words tokens=%s",
            tokens,
        )
        return tokens

    if not raw_text:
        LOGGER.debug("minimal_chat.build_tokens source=raw_text_empty -> []")
        return []

    if " " in raw_text:
        tokens = [part.strip() for part in raw_text.split() if part.strip()]
        LOGGER.debug(
            "minimal_chat.build_tokens source=space_split raw_text=%s tokens=%s",
            raw_text,
            tokens,
        )
        return tokens

    tokens = greedy_tokenize_by_lexicon(raw_text=raw_text, lexicon=lexicon)
    LOGGER.debug(
        "minimal_chat.build_tokens source=greedy_lexicon raw_text=%s tokens=%s",
        raw_text,
        tokens,
    )
    return tokens


def greedy_tokenize_by_lexicon(
    raw_text: str,
    lexicon: LexiconContainer,
) -> List[str]:
    LOGGER.debug("minimal_chat.greedy_tokenize.start raw_text=%s", raw_text)

    words = lexicon.entries.keys()
    length_index = build_length_index(words)

    text = raw_text.strip()
    tokens: List[str] = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        LOGGER.debug(
            "minimal_chat.greedy_tokenize.scan index=%s char=%s remaining=%s",
            i,
            ch,
            text[i:],
        )

        if ch.isspace():
            LOGGER.debug("minimal_chat.greedy_tokenize.skip reason=space index=%s", i)
            i += 1
            continue

        if ch in {"。", "、", "？", "?", "！", "!"}:
            tokens.append(ch)
            LOGGER.debug(
                "minimal_chat.greedy_tokenize.accept_punct token=%s index=%s",
                ch,
                i,
            )
            i += 1
            continue

        matched = longest_match_from_index(
            text=text,
            start=i,
            length_index=length_index,
        )
        if matched:
            tokens.append(matched)
            LOGGER.debug(
                "minimal_chat.greedy_tokenize.accept_match token=%s index=%s next_index=%s",
                matched,
                i,
                i + len(matched),
            )
            i += len(matched)
            continue

        tokens.append(ch)
        LOGGER.debug(
            "minimal_chat.greedy_tokenize.accept_fallback_char token=%s index=%s",
            ch,
            i,
        )
        i += 1

    LOGGER.debug("minimal_chat.greedy_tokenize.result=%s", tokens)
    return tokens


def build_length_index(words: Iterable[str]) -> Dict[int, Set[str]]:
    index: Dict[int, Set[str]] = {}
    count = 0
    for word in words:
        w = str(word).strip()
        if not w:
            LOGGER.debug("minimal_chat.build_length_index.skip reason=empty_word")
            continue
        index.setdefault(len(w), set()).add(w)
        count += 1

    LOGGER.debug(
        "minimal_chat.build_length_index.result total_words=%s lengths=%s",
        count,
        sorted(index.keys()),
    )
    return index


def longest_match_from_index(
    text: str,
    start: int,
    length_index: Dict[int, Set[str]],
) -> str:
    remaining = len(text) - start
    candidate_lengths = sorted(
        (length for length in length_index.keys() if length <= remaining),
        reverse=True,
    )

    LOGGER.debug(
        "minimal_chat.longest_match.start start=%s remaining=%s candidate_lengths=%s slice=%s",
        start,
        remaining,
        candidate_lengths,
        text[start:],
    )

    for length in candidate_lengths:
        piece = text[start : start + length]
        LOGGER.debug(
            "minimal_chat.longest_match.try length=%s piece=%s",
            length,
            piece,
        )
        if piece in length_index[length]:
            LOGGER.debug(
                "minimal_chat.longest_match.hit length=%s piece=%s",
                length,
                piece,
            )
            return piece

    LOGGER.debug("minimal_chat.longest_match.result none")
    return ""


def append_trace_jsonl(trace: TraceLog, trace_dir: Path) -> Path:
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / "latest_trace.jsonl"

    record = dataclass_to_dict(trace)
    LOGGER.debug(
        "minimal_chat.append_trace path=%s record_keys=%s",
        path,
        list(record.keys()) if isinstance(record, dict) else type(record),
    )

    with path.open("a", encoding="utf-8") as f:
        line = json.dumps(record, ensure_ascii=False)
        f.write(line + "\n")
        LOGGER.debug(
            "minimal_chat.append_trace.written bytes=%s",
            len((line + "\n").encode("utf-8")),
        )

    return path


if __name__ == "__main__":
    main()