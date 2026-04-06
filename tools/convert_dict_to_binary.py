from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

from src.core.io.lsd_lexicon import load_lexicon_container, save_lexicon_container


SUPPORTED_FORMATS = ("json", "lsd", "lsdx")


def default_output_path(input_path: Path, fmt: str) -> Path:
    return input_path.with_suffix(f".{fmt}")


def detect_format_from_suffix(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported file suffix: {path.suffix!r} "
            f"(supported: {', '.join(SUPPORTED_FORMATS)})"
        )
    return suffix


def resolve_output_format(output_path: Path, requested_format: str) -> str:
    if requested_format != "auto":
        return requested_format
    return detect_format_from_suffix(output_path)


def normalize_container(container: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(container, dict):
        raise TypeError("Loaded lexicon container must be a dict.")

    meta = container.get("meta", {})
    entries = container.get("entries", {})
    indexes = container.get("indexes", {})

    if not isinstance(meta, dict):
        raise TypeError("container['meta'] must be a dict.")
    if not isinstance(entries, dict):
        raise TypeError("container['entries'] must be a dict.")
    if not isinstance(indexes, dict):
        raise TypeError("container['indexes'] must be a dict.")

    return {
        "meta": meta,
        "entries": entries,
        "indexes": indexes,
        **{k: v for k, v in container.items() if k not in {"meta", "entries", "indexes"}},
    }


def summarize_container(container: Dict[str, Any]) -> Dict[str, Any]:
    meta = container.get("meta", {})
    entries = container.get("entries", {})
    indexes = container.get("indexes", {})

    semantic_axes = meta.get("semantic_axes", [])
    by_pos = indexes.get("by_pos", {})

    return {
        "entry_count": len(entries),
        "semantic_axis_count": len(semantic_axes) if isinstance(semantic_axes, list) else 0,
        "pos_groups": len(by_pos) if isinstance(by_pos, dict) else 0,
    }


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def check_same_entries(
    src_entries: Dict[str, Any],
    dst_entries: Dict[str, Any],
) -> None:
    if len(src_entries) != len(dst_entries):
        raise RuntimeError(
            f"Verification failed: entry count mismatch "
            f"(input={len(src_entries)} output={len(dst_entries)})"
        )

    src_keys = set(src_entries.keys())
    dst_keys = set(dst_entries.keys())
    if src_keys != dst_keys:
        missing = sorted(src_keys - dst_keys)[:10]
        extra = sorted(dst_keys - src_keys)[:10]
        raise RuntimeError(
            "Verification failed: entry keys mismatch "
            f"(missing={missing}, extra={extra})"
        )


def check_meta_compatibility(
    src_meta: Dict[str, Any],
    dst_meta: Dict[str, Any],
) -> None:
    src_axes = src_meta.get("semantic_axes", [])
    dst_axes = dst_meta.get("semantic_axes", [])

    if isinstance(src_axes, list) and isinstance(dst_axes, list):
        if src_axes != dst_axes:
            raise RuntimeError(
                "Verification failed: semantic_axes mismatch "
                f"(input={src_axes}, output={dst_axes})"
            )


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def print_summary(label: str, summary: Dict[str, Any]) -> None:
    print(
        f"[{label}] "
        f"entries={summary['entry_count']} "
        f"semantic_axes={summary['semantic_axis_count']} "
        f"pos_groups={summary['pos_groups']}"
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="dict.json / .lsd / .lsdx を軽量バイナリ辞書へ変換します"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="入力ファイル (.json / .lsd / .lsdx)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="出力先。未指定時は --format に応じて自動決定",
    )
    parser.add_argument(
        "--format",
        choices=["auto", "json", "lsd", "lsdx"],
        default="lsdx",
        help="出力形式。auto の場合は出力拡張子から判定",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="保存後に再ロードして件数・キー・semantic_axes を検証",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="出力ファイルが既に存在していても上書きする",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    input_path: Path = args.input
    if not input_path.exists():
        print(f"[ERROR] input file not found: {input_path}", file=sys.stderr)
        return 1

    requested_format: str = args.format
    tentative_output = args.output or default_output_path(
        input_path,
        "lsdx" if requested_format == "auto" else requested_format,
    )
    output_format = resolve_output_format(tentative_output, requested_format)
    output_path = args.output or default_output_path(input_path, output_format)

    if output_path.exists() and not args.force:
        print(
            f"[ERROR] output file already exists: {output_path}\n"
            f"        overwriteする場合は --force を付けてください。",
            file=sys.stderr,
        )
        return 1

    if input_path.resolve() == output_path.resolve():
        print(
            "[ERROR] input and output paths are the same. "
            "別の出力先を指定してください。",
            file=sys.stderr,
        )
        return 1

    print(f"[INPUT]   {input_path}")
    print(f"[OUTPUT]  {output_path}")
    print(f"[FORMAT]  {output_format}")

    try:
        container = normalize_container(load_lexicon_container(input_path))
        print_summary("INFO", summarize_container(container))

        ensure_parent_dir(output_path)
        save_lexicon_container(output_path, container)

        if args.verify:
            reloaded = normalize_container(load_lexicon_container(output_path))
            check_same_entries(container.get("entries", {}), reloaded.get("entries", {}))
            check_meta_compatibility(container.get("meta", {}), reloaded.get("meta", {}))
            print_summary("VERIFY", summarize_container(reloaded))
            print("[VERIFY] OK")

        try:
            in_size = input_path.stat().st_size
            out_size = output_path.stat().st_size
            ratio = (out_size / in_size) if in_size else 0.0
            print(
                f"[SIZE]    input={in_size:,} bytes ({format_size(in_size)})  "
                f"output={out_size:,} bytes ({format_size(out_size)})  "
                f"ratio={ratio:.3f}"
            )
        except OSError:
            pass

        print("[DONE] conversion completed successfully")
        return 0

    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())