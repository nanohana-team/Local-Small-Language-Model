from __future__ import annotations

import json
import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")


class TraceLogger:
    """Small three-layer logger for the v4 minimal vertical slice."""

    def __init__(
        self,
        runtime_dir: str | Path = "runtime",
        *,
        mode: str = "standard",
        rotate_latest: bool = True,
    ) -> None:
        self.runtime_dir = Path(runtime_dir)
        self.logs_dir = self.runtime_dir / "logs"
        self.traces_dir = self.runtime_dir / "traces"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode

        self.latest_log_path = self.logs_dir / "latest.log"
        self.debug_log_path = self.logs_dir / "debug.log"
        self.latest_trace_path = self.traces_dir / "latest.jsonl"
        if rotate_latest:
            self._rotate_if_needed(self.latest_log_path)
            self._rotate_if_needed(self.debug_log_path, suffix="_debug")
            self._rotate_if_needed(self.latest_trace_path)

        self.session_id = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
        self._turn_counter = 0

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
        if self.mode in {"standard", "deep_trace"}:
            self._write_line(self.debug_log_path, "DEBUG", message)

    def record_trace(self, payload: Mapping[str, Any]) -> None:
        serializable = self._to_jsonable(dict(payload))
        serializable.setdefault("session_id", self.session_id)
        serializable.setdefault("timestamp_jst", datetime.now(JST).isoformat())
        with self.latest_trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(serializable, ensure_ascii=False) + "\n")

    def _write_line(self, path: Path, level: str, message: str) -> None:
        timestamp = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp} [{level}] {message}\n")

    def _rotate_if_needed(self, latest_path: Path, suffix: str = "") -> None:
        if not latest_path.exists() or latest_path.stat().st_size == 0:
            return
        stamp = datetime.now(JST).strftime("%Y%m%d%H%M%S")
        archive_name = f"{stamp}{suffix}{latest_path.suffix}"
        archive_path = latest_path.with_name(archive_name)
        counter = 1
        while archive_path.exists():
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


__all__ = ["TraceLogger"]
