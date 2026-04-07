from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from src.core.io.lsd_lexicon import load_lexicon_container
from src.core.schema import LexiconContainer, UnknownSpan
from src.training.contamination_guard import assess_dict_entry
from src.training.unknown_word_learner import UnknownWordLearner, build_unknown_word_learner_config
from src.utils.settings import get_setting, load_settings


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description='Re-evaluate quarantined dictionary candidates and promote safe entries.')
    parser.add_argument('--settings-dir', default='settings')
    parser.add_argument('--lexicon', default=None)
    parser.add_argument('--quarantine-path', default=None)
    parser.add_argument('--limit', type=int, default=100)
    args = parser.parse_args()

    settings = load_settings(args.settings_dir)
    lexicon_path = str(args.lexicon or get_setting(settings, 'paths', 'lexicon', default='libs/dict.lsdx'))
    learner_cfg = build_unknown_word_learner_config(get_setting(settings, 'pipeline', 'unknown_word', default={}))
    if args.quarantine_path:
        learner_cfg.quarantine_path = args.quarantine_path

    container_raw = load_lexicon_container(lexicon_path)
    lexicon = LexiconContainer.from_dict(container_raw)
    learner = UnknownWordLearner(lexicon=lexicon, config=learner_cfg)

    quarantine_rows = _load_jsonl(Path(learner_cfg.quarantine_path))
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in quarantine_rows:
        surface = str(row.get('surface', '') or row.get('word', '')).strip()
        if surface:
            grouped[surface].append(row)

    promoted = 0
    reviewed = 0
    for surface, rows in list(grouped.items())[: max(1, int(args.limit))]:
        row = rows[-1]
        entry = dict(row.get('entry', {}) or {})
        occurrence_count = max(int(row.get('occurrence_count', len(rows)) or len(rows)), len(rows))
        assessment = assess_dict_entry(
            word=str(entry.get('word', surface) or surface),
            entry_payload=entry,
            surface=surface,
            occurrence_count=occurrence_count,
            min_overlay_confidence=float(learner_cfg.min_overlay_confidence),
            watch_threshold=float(learner_cfg.contamination_watch_threshold),
            danger_threshold=float(learner_cfg.contamination_danger_threshold),
        )
        reviewed += 1
        if assessment.status != 'safe':
            span = UnknownSpan(
                surface=surface,
                start=int(row.get('start', 0) or 0),
                end=int(row.get('end', len(surface)) or len(surface)),
                reason=str(row.get('reason', 'quarantine_relearn') or 'quarantine_relearn'),
                pos_hint=str(row.get('pos_hint', '') or ''),
                proper_noun_candidate=bool(row.get('proper_noun_candidate', False)),
                named_entity_type_hint=str(row.get('named_entity_type_hint', '') or ''),
            )
            entry = learner._request_relearned_entry_payload(  # noqa: SLF001
                surface=surface,
                raw_text=str(row.get('raw_text', surface) or surface),
                span=span,
                previous_payload=entry,
                risk_reasons=assessment.reasons,
            )
            assessment = assess_dict_entry(
                word=str(entry.get('word', surface) or surface),
                entry_payload=entry,
                surface=surface,
                occurrence_count=occurrence_count,
                min_overlay_confidence=float(learner_cfg.min_overlay_confidence),
                watch_threshold=float(learner_cfg.contamination_watch_threshold),
                danger_threshold=float(learner_cfg.contamination_danger_threshold),
            )
        if assessment.status == 'safe' and occurrence_count >= int(learner_cfg.promote_threshold):
            word = str(entry.get('word', surface) or surface)
            if learner._entry_allowed_for_overlay(word, entry, source='quarantine_relearn'):  # noqa: SLF001
                learner._save_overlay_entry(word, entry, risk_assessment=assessment.to_dict(), occurrence_count=occurrence_count)  # noqa: SLF001
                promoted += 1

    print(json.dumps({
        'reviewed': reviewed,
        'promoted': promoted,
        'quarantine_path': learner_cfg.quarantine_path,
        'overlay_path': learner_cfg.overlay_path,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
