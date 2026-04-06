from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Merge runtime lexicon overlay into dict.json')
    parser.add_argument('base_dict', help='Base dict.json path')
    parser.add_argument('--overlay', default='runtime/lexicon_overlay.json', help='Overlay JSON path')
    parser.add_argument('-o', '--output', default='', help='Output dict.json path (default: overwrite base)')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_dict)
    overlay_path = Path(args.overlay)
    output_path = Path(args.output) if args.output else base_path

    if not base_path.exists():
        raise FileNotFoundError(f'Base dict file not found: {base_path}')
    if not overlay_path.exists():
        raise FileNotFoundError(f'Overlay file not found: {overlay_path}')

    base_payload = json.loads(base_path.read_text(encoding='utf-8'))
    overlay_payload = json.loads(overlay_path.read_text(encoding='utf-8'))

    if not isinstance(base_payload, dict):
        raise TypeError('Base dict.json must be a JSON object.')
    if not isinstance(overlay_payload, dict):
        raise TypeError('Overlay JSON must be a JSON object.')

    base_entries: Dict[str, Any] = dict(base_payload.get('entries', {}) or {})
    overlay_entries: Dict[str, Any] = dict(overlay_payload.get('entries', {}) or {})

    merged_count = 0
    for word, entry in overlay_entries.items():
        base_entries[str(word)] = entry
        merged_count += 1

    base_payload['entries'] = base_entries
    meta = dict(base_payload.get('meta', {}) or {})
    meta['entry_count'] = len(base_entries)
    base_payload['meta'] = meta

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(base_payload, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'[MERGED] overlay_entries={merged_count}')
    print(f'[OUTPUT] {output_path}')


if __name__ == '__main__':
    main()
