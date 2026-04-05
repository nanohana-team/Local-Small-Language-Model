from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core.io.lsd_lexicon import load_lexicon_container, save_lexicon_container


def default_output_path(input_path: Path, fmt: str) -> Path:
    if fmt == "lsd":
        return input_path.with_suffix(".lsd")
    if fmt == "lsdx":
        return input_path.with_suffix(".lsdx")
    return input_path.with_suffix(".json")


def main() -> None:
    parser = argparse.ArgumentParser(description="dict.json を独自バイナリ形式へ変換します")
    parser.add_argument("input", type=Path, help="入力 dict.json / .lsd / .lsdx")
    parser.add_argument("-o", "--output", type=Path, default=None, help="出力先")
    parser.add_argument("--format", choices=["lsd", "lsdx", "json"], default="lsdx", help="出力形式")
    parser.add_argument("--verify", action="store_true", help="変換後に再ロードして件数検証")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or default_output_path(input_path, args.format)

    print(f"[INPUT]  {input_path}")
    print(f"[OUTPUT] {output_path}")
    print(f"[FORMAT] {args.format}")

    container = load_lexicon_container(input_path)
    entries = container.get("entries", {})
    meta = container.get("meta", {})
    print(f"[INFO] entries={len(entries)} semantic_axes={len(meta.get('semantic_axes', []))}")

    save_lexicon_container(output_path, container)

    if args.verify:
        reloaded = load_lexicon_container(output_path)
        reloaded_entries = reloaded.get("entries", {})
        if len(reloaded_entries) != len(entries):
            raise RuntimeError(
                f"Verification failed: input={len(entries)} output={len(reloaded_entries)}"
            )
        print(f"[VERIFY] OK entries={len(reloaded_entries)}")

    try:
        in_size = input_path.stat().st_size
        out_size = output_path.stat().st_size
        ratio = (out_size / in_size) if in_size else 0.0
        print(f"[SIZE] input={in_size:,} bytes output={out_size:,} bytes ratio={ratio:.3f}")
    except OSError:
        pass


if __name__ == "__main__":
    main()
