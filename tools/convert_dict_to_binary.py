from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.io.lsd_lexicon import (
    collect_lexicon_validation_report,
    collect_raw_lexicon_validation_report,
    load_lexicon_container,
    normalize_lexicon_container,
    save_lexicon_container,
    stable_json_dumps,
)


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


def load_input_containers(input_path: Path) -> tuple[Dict[str, Any] | None, Dict[str, Any]]:
    suffix = input_path.suffix.lower()
    raw_container: Dict[str, Any] | None = None
    if suffix == ".json":
        with input_path.open("r", encoding="utf-8") as f:
            raw_value = json.load(f)
        if not isinstance(raw_value, dict):
            raise TypeError("Unsupported JSON lexicon format")
        raw_container = dict(raw_value)
        normalized = normalize_container(normalize_lexicon_container(raw_container))
        return raw_container, normalized
    return None, normalize_container(load_lexicon_container(input_path))


def print_validation_report(label: str, report: Dict[str, Any]) -> None:
    print(
        f"[{label}] errors={report.get('error_count', 0)} "
        f"warnings={report.get('warning_count', 0)}"
    )
    for warning in report.get("warnings", [])[:10]:
        print(f"[{label}][WARNING] {warning}")
    remaining = int(report.get("warning_count", 0)) - min(len(report.get("warnings", [])), 10)
    if remaining > 0:
        print(f"[{label}][WARNING] ... and {remaining} more warnings")


def summarize_container(container: Dict[str, Any]) -> Dict[str, Any]:
    meta = container.get("meta", {})
    entries = container.get("entries", {})
    indexes = container.get("indexes", {})

    semantic_axes = meta.get("semantic_axes", [])
    by_pos = indexes.get("by_pos", {})
    concepts = container.get("concepts", {})
    slot_frames = container.get("slot_frames", {})

    bridge_entry_count = 0
    concept_link_count = 0
    sense_count = 0
    surface_form_count = 0
    relation_count = 0
    dangling_relation_count = 0
    if isinstance(concepts, dict):
        for concept_id, concept in concepts.items():
            if not isinstance(concept, dict):
                continue
            relations = concept.get("relations", [])
            if not isinstance(relations, list):
                continue
            relation_count += len(relations)
            for relation in relations:
                if not isinstance(relation, dict):
                    continue
                target = relation.get("target")
                if isinstance(target, str) and target and target not in concepts:
                    dangling_relation_count += 1
    for entry in entries.values():
        if not isinstance(entry, dict):
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

    return {
        "entry_count": len(entries),
        "semantic_axis_count": len(semantic_axes) if isinstance(semantic_axes, list) else 0,
        "pos_groups": len(by_pos) if isinstance(by_pos, dict) else 0,
        "concept_count": len(concepts) if isinstance(concepts, dict) else 0,
        "slot_frame_count": len(slot_frames) if isinstance(slot_frames, dict) else 0,
        "bridge_entry_count": bridge_entry_count,
        "concept_link_count": concept_link_count,
        "sense_count": sense_count,
        "surface_form_count": surface_form_count,
        "relation_count": relation_count,
        "dangling_relation_count": dangling_relation_count,
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


def check_bridge_field_compatibility(
    src_entries: Dict[str, Any],
    dst_entries: Dict[str, Any],
) -> None:
    bridge_fields = ("surface_forms", "senses", "concept_ids", "slot_frame_id", "reading")
    for key in src_entries.keys():
        src_entry = src_entries.get(key, {})
        dst_entry = dst_entries.get(key, {})
        if not isinstance(src_entry, dict) or not isinstance(dst_entry, dict):
            continue
        for field in bridge_fields:
            src_value = src_entry.get(field)
            dst_value = dst_entry.get(field)
            if stable_json_dumps(src_value) != stable_json_dumps(dst_value):
                raise RuntimeError(
                    "Verification failed: bridge field mismatch "
                    f"(entry={key!r}, field={field!r}, input={src_value!r}, output={dst_value!r})"
                )


def check_top_level_compatibility(
    src_container: Dict[str, Any],
    dst_container: Dict[str, Any],
) -> None:
    for field in ("concepts", "slot_frames"):
        src_value = src_container.get(field, {})
        dst_value = dst_container.get(field, {})
        if stable_json_dumps(src_value) != stable_json_dumps(dst_value):
            raise RuntimeError(
                "Verification failed: top-level field mismatch "
                f"(field={field!r})"
            )


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def print_summary(label: str, summary: Dict[str, Any]) -> None:
    print(
        f"[{label}] "
        f"entries={summary['entry_count']} "
        f"semantic_axes={summary['semantic_axis_count']} "
        f"pos_groups={summary['pos_groups']} "
        f"bridge_entries={summary['bridge_entry_count']} "
        f"concept_links={summary['concept_link_count']} "
        f"senses={summary['sense_count']} "
        f"surface_forms={summary['surface_form_count']} "
        f"relations={summary['relation_count']} "
        f"dangling_relations={summary['dangling_relation_count']} "
        f"concepts={summary['concept_count']} "
        f"slot_frames={summary['slot_frame_count']}"
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
        "--validate-only",
        action="store_true",
        help="入力辞書の検証だけを行い、変換出力は行わない",
    )
    parser.add_argument(
        "--strict-schema",
        action="store_true",
        help="v4 の最小正式スキーマ契約として厳格に検証する",
    )
    parser.add_argument(
        "--strict-relations",
        action="store_true",
        help="relation type / direction / usage_stage / source などの契約を追加で厳格検証する",
    )
    parser.add_argument(
        "--require-closed-relations",
        action="store_true",
        help="存在しない target を参照する relation を warning ではなく error として扱う",
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
    actual_output_format = detect_format_from_suffix(output_path)

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
                "[ERROR] input and output paths are the same. "
                "別の出力先を指定してください。",
                file=sys.stderr,
            )
            return 1

    print(f"[INPUT]   {input_path}")
    if args.validate_only:
        print("[MODE]    validate-only")
    else:
        print(f"[OUTPUT]  {output_path}")
        print(f"[FORMAT]  {actual_output_format}")

    try:
        raw_container, container = load_input_containers(input_path)
        print_summary("INFO", summarize_container(container))

        if raw_container is not None:
            validation = collect_raw_lexicon_validation_report(
                raw_container,
                strict_schema=args.strict_schema,
                strict_relations=args.strict_relations,
                require_closed_relations=args.require_closed_relations,
            )
        else:
            validation = collect_lexicon_validation_report(
                container,
                strict_schema=args.strict_schema,
                strict_relations=args.strict_relations,
                require_closed_relations=args.require_closed_relations,
            )

        print_validation_report("VALIDATE", validation)
        if validation.get("error_count", 0):
            messages = validation.get("errors", [])
            preview = "\n".join(f"- {message}" for message in messages[:20])
            if len(messages) > 20:
                preview += f"\n- ... and {len(messages) - 20} more"
            raise RuntimeError(f"Validation failed:\n{preview}")

        if args.validate_only:
            print("[DONE] validation completed successfully")
            return 0

        ensure_parent_dir(output_path)
        save_lexicon_container(output_path, container)

        if args.verify:
            reloaded = normalize_container(load_lexicon_container(output_path))
            check_same_entries(container.get("entries", {}), reloaded.get("entries", {}))
            check_meta_compatibility(container.get("meta", {}), reloaded.get("meta", {}))
            check_bridge_field_compatibility(container.get("entries", {}), reloaded.get("entries", {}))
            check_top_level_compatibility(container, reloaded)
            verify_validation = collect_lexicon_validation_report(
                reloaded,
                strict_schema=args.strict_schema,
                strict_relations=args.strict_relations,
                require_closed_relations=args.require_closed_relations,
            )
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

        print("[DONE] conversion completed successfully")
        return 0

    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())