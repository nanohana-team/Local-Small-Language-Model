from __future__ import annotations

import json
import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.core.scoring import load_scoring_config

try:
    JST = ZoneInfo("Asia/Tokyo")
except ZoneInfoNotFoundError:
    JST = timezone(timedelta(hours=9))


DEFAULT_TRACE_LIMITS: Dict[str, Dict[str, int]] = {
    "minimal": {
        "divergence_candidates": 4,
        "seed_matches": 4,
        "explored_relations": 0,
        "convergence_candidates": 4,
        "rejected_candidates": 0,
        "accepted_relations": 6,
        "rejected_relations": 0,
    },
    "standard": {
        "divergence_candidates": 10,
        "seed_matches": 8,
        "explored_relations": 8,
        "convergence_candidates": 6,
        "rejected_candidates": 4,
        "accepted_relations": 12,
        "rejected_relations": 8,
    },
    "deep_trace": {
        "divergence_candidates": -1,
        "seed_matches": -1,
        "explored_relations": -1,
        "convergence_candidates": -1,
        "rejected_candidates": -1,
        "accepted_relations": -1,
        "rejected_relations": -1,
    },
}


class TraceLogger:
    """Small three-layer logger for the v4 minimal vertical slice."""

    def __init__(
        self,
        runtime_dir: str | Path = "runtime",
        *,
        mode: str = "standard",
        rotate_latest: bool = True,
        scoring_config_path: str | None = None,
        debug_enabled: bool = False,
    ) -> None:
        self.runtime_dir = Path(runtime_dir)
        self.logs_dir = self.runtime_dir / "logs"
        self.traces_dir = self.runtime_dir / "traces"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.scoring_config_path = scoring_config_path
        self.debug_enabled = debug_enabled or mode == "deep_trace"
        self.trace_limits = self._load_trace_limits(scoring_config_path)
        self.scoring_config, self.resolved_scoring_config_path = load_scoring_config(scoring_config_path)

        self.latest_log_path = self.logs_dir / "latest.log"
        self.debug_log_path = self.logs_dir / "debug.log"
        self.latest_trace_path = self.traces_dir / "latest.jsonl"
        self.latest_manifest_path = self.traces_dir / "latest.session.json"
        if rotate_latest:
            self._rotate_if_needed(self.latest_log_path)
            self._rotate_if_needed(self.debug_log_path, suffix="_debug")
            self._rotate_if_needed(self.latest_trace_path)
            self._rotate_if_needed(self.latest_manifest_path, suffix="_session", preserve_json=True)

        self.session_id = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
        self.session_manifest_archive_path = self.traces_dir / f"{self.session_id}.session.json"
        self._turn_counter = 0
        self._session_manifest: Dict[str, Any] = {
            "schema_version": "trace_session_manifest_v1",
            "record_type": "trace_session_manifest",
            "session_id": self.session_id,
            "timestamp_jst": datetime.now(JST).isoformat(),
            "trace_mode": self.mode,
            "debug_enabled": bool(self.debug_enabled),
            "runtime_paths": {
                "logs_latest": str(self.latest_log_path),
                "debug_log": str(self.debug_log_path),
                "traces_latest": str(self.latest_trace_path),
            },
            "scoring": {
                "config_path": self.resolved_scoring_config_path,
                "config_version": self.scoring_config.get("version", 1),
                "weights": self.scoring_config.get("weights", {}),
                "thresholds": self.scoring_config.get("thresholds", {}),
                "trace_limits": self.trace_limits.get(self.mode, {}),
            },
        }
        self._write_session_manifest()

    def next_turn_id(self) -> str:
        self._turn_counter += 1
        stamp = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
        return f"{stamp}_{self._turn_counter:04d}"

    def info(self, message: str) -> None:
        self._write_line(self.latest_log_path, "INFO", message)

    def warning(self, message: str) -> None:
        self._write_line(self.latest_log_path, "WARNING", message)

    def error(self, message: str) -> None:
        self._write_line(self.latest_log_path, "ERROR", message)
        self._write_line(self.debug_log_path, "ERROR", message)

    def debug(self, message: str) -> None:
        if self.debug_enabled:
            self._write_line(self.debug_log_path, "DEBUG", message)

    def update_session_manifest(self, **fields: Any) -> None:
        for key, value in fields.items():
            if value is None:
                continue
            if isinstance(value, Mapping) and isinstance(self._session_manifest.get(key), Mapping):
                merged = dict(self._session_manifest.get(key, {}))
                merged.update(self._to_jsonable(value))
                self._session_manifest[key] = merged
            else:
                self._session_manifest[key] = self._to_jsonable(value)
        self._write_session_manifest()

    def record_trace(self, payload: Mapping[str, Any]) -> None:
        serializable = self._to_jsonable(dict(payload))
        serializable.setdefault("schema_version", "trace_v1")
        serializable.setdefault("record_type", "trace")
        serializable.setdefault("session_id", self.session_id)
        serializable.setdefault("timestamp_jst", datetime.now(JST).isoformat())
        serializable["trace_mode"] = self.mode
        shaped = self._shape_trace_payload(serializable)
        with self.latest_trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(shaped, ensure_ascii=False) + "\n")

    def _write_session_manifest(self) -> None:
        manifest = self._to_jsonable(self._session_manifest)
        text = json.dumps(manifest, ensure_ascii=False, indent=2)
        self.latest_manifest_path.write_text(text + "\n", encoding="utf-8")
        self.session_manifest_archive_path.write_text(text + "\n", encoding="utf-8")

    def _write_line(self, path: Path, level: str, message: str) -> None:
        timestamp = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp} [{level}] {message}\n")

    def _rotate_if_needed(self, latest_path: Path, suffix: str = "", preserve_json: bool = False) -> None:
        if not latest_path.exists() or latest_path.stat().st_size == 0:
            return
        stamp = datetime.now(JST).strftime("%Y%m%d%H%M%S")
        if preserve_json:
            archive_name = f"{stamp}{suffix}.json"
        else:
            archive_name = f"{stamp}{suffix}{latest_path.suffix}"
        archive_path = latest_path.with_name(archive_name)
        counter = 1
        while archive_path.exists():
            if preserve_json:
                archive_path = latest_path.with_name(f"{stamp}{suffix}_{counter}.json")
            else:
                archive_path = latest_path.with_name(f"{stamp}{suffix}_{counter}{latest_path.suffix}")
            counter += 1
        shutil.move(str(latest_path), str(archive_path))

    def _to_jsonable(self, value: Any) -> Any:
        if is_dataclass(value):
            return self._to_jsonable(asdict(value))
        if isinstance(value, Mapping):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(v) for v in value]
        if isinstance(value, set):
            return [self._to_jsonable(v) for v in sorted(value, key=str)]
        return value

    def _load_trace_limits(self, scoring_config_path: str | None) -> Dict[str, Dict[str, int]]:
        config, _ = load_scoring_config(scoring_config_path)
        trace_config = config.get("trace") if isinstance(config.get("trace"), Mapping) else {}
        merged = {mode: dict(values) for mode, values in DEFAULT_TRACE_LIMITS.items()}
        for mode, values in trace_config.items():
            if mode not in merged or not isinstance(values, Mapping):
                continue
            for key, value in values.items():
                try:
                    merged[mode][str(key)] = int(value)
                except (TypeError, ValueError):
                    continue
        return merged

    def _limit_value(self, mode: str, key: str) -> int:
        return int(self.trace_limits.get(mode, {}).get(key, DEFAULT_TRACE_LIMITS.get(mode, {}).get(key, -1)))

    def _trim_sequence(self, value: Any, limit: int) -> Any:
        if not isinstance(value, list):
            return value
        if limit < 0:
            return value
        return value[:limit]

    def _relation_type_counts(self, relations: list[Any]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for relation in relations:
            if not isinstance(relation, Mapping):
                continue
            relation_type = str(relation.get("type") or "unknown")
            counts[relation_type] = counts.get(relation_type, 0) + 1
        return counts

    def _summarize_explored_relations(self, relations: list[Any], *, mode: str) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "count": len(relations),
            "type_counts": self._relation_type_counts(relations),
            "max_depth": max((int(item.get("depth") or 0) for item in relations if isinstance(item, Mapping)), default=0),
        }
        sample_limit = 0 if mode == "minimal" else self._limit_value(mode, "explored_relations")
        if sample_limit != 0:
            sample = []
            for relation in relations[: max(0, sample_limit)]:
                if not isinstance(relation, Mapping):
                    continue
                sample.append(
                    {
                        "from": relation.get("from"),
                        "type": relation.get("type"),
                        "to": relation.get("to"),
                        "depth": relation.get("depth"),
                    }
                )
            if sample:
                summary["sample"] = sample
        return summary

    def _summarize_slot_evidence(self, slot_evidence: Mapping[str, Any], *, mode: str) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        if not isinstance(slot_evidence, Mapping):
            return summary
        for key, value in slot_evidence.items():
            if value in (None, [], {}, ""):
                continue
            if isinstance(value, list):
                if mode == "deep_trace":
                    summary[str(key)] = value
                else:
                    summary[str(key)] = {
                        "count": len(value),
                        "sample": value[:3],
                    }
            elif isinstance(value, Mapping):
                summary[str(key)] = {
                    "keys": list(value.keys())[:6],
                    "count": len(value),
                }
            else:
                summary[str(key)] = value
        return summary

    def _shape_trace_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        mode = self.mode if self.mode in DEFAULT_TRACE_LIMITS else "standard"
        shaped = dict(payload)

        raw_divergence_candidates = list(payload.get("divergence_candidates") or [])
        raw_seed_matches = list(payload.get("seed_matches") or [])
        raw_explored_relations = list(payload.get("explored_relations") or [])
        raw_convergence_candidates = list(payload.get("convergence_candidates") or [])
        raw_rejected_candidates = list(payload.get("rejected_candidates") or [])
        raw_accepted_relations = list(payload.get("accepted_relations") or [])
        raw_rejected_relations = list(payload.get("rejected_relations") or [])

        shaped["divergence_candidates"] = self._trim_sequence(raw_divergence_candidates, self._limit_value(mode, "divergence_candidates"))
        shaped["seed_matches"] = self._trim_sequence(raw_seed_matches, self._limit_value(mode, "seed_matches"))
        shaped["convergence_candidates"] = self._trim_sequence(raw_convergence_candidates, self._limit_value(mode, "convergence_candidates"))
        shaped["rejected_candidates"] = self._trim_sequence(raw_rejected_candidates, self._limit_value(mode, "rejected_candidates"))
        shaped["accepted_relations"] = self._trim_sequence(raw_accepted_relations, self._limit_value(mode, "accepted_relations"))
        shaped["rejected_relations"] = self._trim_sequence(raw_rejected_relations, self._limit_value(mode, "rejected_relations"))

        if mode == "deep_trace":
            shaped["explored_relations"] = raw_explored_relations
        else:
            shaped["explored_relations"] = self._summarize_explored_relations(raw_explored_relations, mode=mode)

        if "slot_evidence" in shaped and mode != "deep_trace":
            shaped["slot_evidence"] = self._summarize_slot_evidence(shaped.get("slot_evidence") or {}, mode=mode)

        scoring_details = dict(shaped.get("scoring_details") or {})
        if scoring_details:
            scoring_details.pop("weights", None)
            scoring_details.pop("thresholds", None)
            scoring_details.pop("config_path", None)
            shaped["scoring_details"] = scoring_details

        shaped.pop("startup", None)

        shaped["trace_summary"] = {
            "seed_count": len(raw_seed_matches),
            "divergence_candidate_count": len(raw_divergence_candidates),
            "explored_relation_count": len(raw_explored_relations),
            "accepted_relation_count": len(raw_accepted_relations),
            "rejected_relation_count": len(raw_rejected_relations),
            "missing_slot_count": len(shaped.get("missing_slots") or []),
            "total_ms": ((shaped.get("timing") or {}).get("total_ms")),
        }

        if mode == "minimal":
            shaped.pop("rejected_candidates", None)
            shaped.pop("rejected_relations", None)
            shaped.pop("slot_evidence", None)
        return shaped


__all__ = ["TraceLogger"]
