from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.io.lsd_lexicon import (
    collect_lexicon_validation_report,
    collect_raw_lexicon_validation_report,
    export_entries_lexicon_container,
    export_hierarchical_lexicon_container,
    export_lexical_entries_lexicon_container,
    load_lexicon_container,
    normalize_lexicon_container,
    stable_json_dumps,
)


EXPORT_STYLES = ("lexical", "entries", "hierarchical")


def default_output_path(input_path: Path) -> Path:
    return input_path.with_suffix(".json")


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def summarize_container(container: Mapping[str, Any]) -> Dict[str, Any]:
    meta = container.get("meta", {}) if isinstance(container.get("meta"), Mapping) else {}
    entries = container.get("entries", {}) if isinstance(container.get("entries"), Mapping) else {}
    indexes = container.get("indexes", {}) if isinstance(container.get("indexes"), Mapping) else {}
    concepts = container.get("concepts", {}) if isinstance(container.get("concepts"), Mapping) else {}
    slot_frames = container.get("slot_frames", {}) if isinstance(container.get("slot_frames"), Mapping) else {}

    bridge_entry_count = 0
    concept_link_count = 0
    sense_count = 0
    surface_form_count = 0
    for entry in entries.values():
        if not isinstance(entry, Mapping):
            continue
        concept_ids = entry.get("concept_ids", [])
        senses = entry.get("senses", [])
        surface_forms = entry.get("surface_forms", [])
        slot_frame_id = entry.get("slot_frame_id")
        reading = entry.get("reading")
        has_explicit_bridge = bool(concept_ids or senses or slot_frame_id or reading)
        if isinstance(surface_forms, list) and len(surface_forms) > 1:
            has_explicit_bridge = True
        if has_explicit_bridge:
            bridge_entry_count += 1
        if isinstance(concept_ids, list):
            concept_link_count += len(concept_ids)
        if isinstance(senses, list):
            sense_count += len(senses)
        if isinstance(surface_forms, list):
            surface_form_count += len(surface_forms)

    semantic_axes = meta.get("semantic_axes", [])
    by_pos = indexes.get("by_pos", {}) if isinstance(indexes.get("by_pos"), Mapping) else {}

    return {
        "entry_count": len(entries),
        "semantic_axis_count": len(semantic_axes) if isinstance(semantic_axes, list) else 0,
        "pos_groups": len(by_pos),
        "bridge_entry_count": bridge_entry_count,
        "concept_link_count": concept_link_count,
        "sense_count": sense_count,
        "surface_form_count": surface_form_count,
        "concept_count": len(concepts),
        "slot_frame_count": len(slot_frames),
    }


def print_summary(label: str, summary: Mapping[str, Any]) -> None:
    print(
        f"[{label}] "
        f"entries={summary['entry_count']} "
        f"semantic_axes={summary['semantic_axis_count']} "
        f"pos_groups={summary['pos_groups']} "
        f"bridge_entries={summary['bridge_entry_count']} "
        f"concept_links={summary['concept_link_count']} "
        f"senses={summary['sense_count']} "
        f"surface_forms={summary['surface_form_count']} "
        f"concepts={summary['concept_count']} "
        f"slot_frames={summary['slot_frame_count']}"
    )


def print_validation_report(label: str, report: Mapping[str, Any]) -> None:
    print(
        f"[{label}] errors={report.get('error_count', 0)} "
        f"warnings={report.get('warning_count', 0)}"
    )
    for warning in report.get("warnings", [])[:10]:
        print(f"[{label}][WARNING] {warning}")
    remaining = int(report.get("warning_count", 0)) - min(len(report.get("warnings", [])), 10)
    if remaining > 0:
        print(f"[{label}][WARNING] ... and {remaining} more warnings")


def check_same_entries(src_entries: Mapping[str, Any], dst_entries: Mapping[str, Any]) -> None:
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


def check_meta_compatibility(src_meta: Mapping[str, Any], dst_meta: Mapping[str, Any]) -> None:
    src_axes = src_meta.get("semantic_axes", [])
    dst_axes = dst_meta.get("semantic_axes", [])
    if isinstance(src_axes, list) and isinstance(dst_axes, list) and src_axes != dst_axes:
        raise RuntimeError(
            "Verification failed: semantic_axes mismatch "
            f"(input={src_axes}, output={dst_axes})"
        )


def check_bridge_field_compatibility(src_entries: Mapping[str, Any], dst_entries: Mapping[str, Any]) -> None:
    bridge_fields = ("surface_forms", "senses", "concept_ids", "slot_frame_id", "reading")
    for key in src_entries.keys():
        src_entry = src_entries.get(key, {})
        dst_entry = dst_entries.get(key, {})
        if not isinstance(src_entry, Mapping) or not isinstance(dst_entry, Mapping):
            continue
        for field in bridge_fields:
            src_value = src_entry.get(field)
            dst_value = dst_entry.get(field)
            if stable_json_dumps(src_value) != stable_json_dumps(dst_value):
                raise RuntimeError(
                    "Verification failed: bridge field mismatch "
                    f"(entry={key!r}, field={field!r}, input={src_value!r}, output={dst_value!r})"
                )


def check_top_level_compatibility(src_container: Mapping[str, Any], dst_container: Mapping[str, Any]) -> None:
    for field in ("concepts", "slot_frames"):
        src_value = src_container.get(field, {})
        dst_value = dst_container.get(field, {})
        if stable_json_dumps(src_value) != stable_json_dumps(dst_value):
            raise RuntimeError(
                "Verification failed: top-level field mismatch "
                f"(field={field!r})"
            )


def export_container(container: Mapping[str, Any], style: str) -> Dict[str, Any]:
    normalized = normalize_lexicon_container(container)
    if style == "lexical":
        return export_lexical_entries_lexicon_container(normalized)
    if style == "entries":
        return export_entries_lexicon_container(normalized)
    if style == "hierarchical":
        return export_hierarchical_lexicon_container(normalized)
    raise ValueError(f"Unsupported export style: {style}")


def validate_exported_container(exported: Mapping[str, Any], style: str, *, strict_schema: bool) -> Dict[str, Any]:
    if style == "lexical":
        return collect_raw_lexicon_validation_report(exported, strict_schema=strict_schema)
    normalized = normalize_lexicon_container(exported)
    return collect_lexicon_validation_report(normalized, strict_schema=strict_schema)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=".lsd / .lsdx を JSON 辞書へ戻します"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="入力ファイル (.lsd / .lsdx / .json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="出力先 JSON。未指定時は input と同名の .json を作成",
    )
    parser.add_argument(
        "--style",
        choices=list(EXPORT_STYLES),
        default="lexical",
        help="出力 JSON の形式。lexical=v4 lexical_entries, entries=正規化 entries, hierarchical=階層 lexicon",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="保存後に再ロードして entry 件数・bridge field・top-level fields を検証",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="入力辞書の読込・検証だけを行い、JSON 出力は行わない",
    )
    parser.add_argument(
        "--strict-schema",
        action="store_true",
        help="lexical style の場合は raw v4 schema、その他は normalized schema として厳格検証する",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="出力 JSON が既に存在していても上書きする",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    input_path: Path = args.input
    if not input_path.exists():
        print(f"[ERROR] input file not found: {input_path}", file=sys.stderr)
        return 1

    output_path = args.output or default_output_path(input_path)
    if output_path.suffix.lower() != ".json":
        print("[ERROR] output must be a .json file", file=sys.stderr)
        return 1

    if not args.validate_only:
        if output_path.exists() and not args.force:
            print(
                f"[ERROR] output file already exists: {output_path}\n"
                f"        overwriteする場合は --force を付けてください。",
                file=sys.stderr,
            )
            return 1
        if input_path.resolve() == output_path.resolve():
            print(
                "[ERROR] input and output paths are the same. 別の出力先を指定してください。",
                file=sys.stderr,
            )
            return 1

    print(f"[INPUT]   {input_path}")
    print(f"[STYLE]   {args.style}")
    if args.validate_only:
        print("[MODE]    validate-only")
    else:
        print(f"[OUTPUT]  {output_path}")

    try:
        container = normalize_lexicon_container(load_lexicon_container(input_path))
        print_summary("INFO", summarize_container(container))
        input_validation = collect_lexicon_validation_report(container, strict_schema=args.strict_schema)
        print_validation_report("VALIDATE", input_validation)
        if input_validation.get("error_count", 0):
            messages = input_validation.get("errors", [])
            preview = "\n".join(f"- {message}" for message in messages[:20])
            if len(messages) > 20:
                preview += f"\n- ... and {len(messages) - 20} more"
            raise RuntimeError(f"Input validation failed:\n{preview}")

        if args.validate_only:
            print("[DONE] validation completed successfully")
            return 0

        exported = export_container(container, args.style)
        output_validation = validate_exported_container(exported, args.style, strict_schema=args.strict_schema)
        print_validation_report("EXPORT", output_validation)
        if output_validation.get("error_count", 0):
            messages = output_validation.get("errors", [])
            preview = "\n".join(f"- {message}" for message in messages[:20])
            if len(messages) > 20:
                preview += f"\n- ... and {len(messages) - 20} more"
            raise RuntimeError(f"Export validation failed:\n{preview}")

        ensure_parent_dir(output_path)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(exported, f, ensure_ascii=False, indent=2)

        if args.verify:
            reloaded = normalize_lexicon_container(load_lexicon_container(output_path))
            check_same_entries(container.get("entries", {}), reloaded.get("entries", {}))
            check_meta_compatibility(container.get("meta", {}), reloaded.get("meta", {}))
            check_bridge_field_compatibility(container.get("entries", {}), reloaded.get("entries", {}))
            check_top_level_compatibility(container, reloaded)
            verify_validation = collect_lexicon_validation_report(reloaded, strict_schema=args.strict_schema)
            print_summary("VERIFY", summarize_container(reloaded))
            print_validation_report("VERIFY", verify_validation)
            if verify_validation.get("error_count", 0):
                raise RuntimeError("Verification reload produced schema errors")
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

        print("[DONE] export completed successfully")
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
