from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    JST = ZoneInfo("Asia/Tokyo")
except ZoneInfoNotFoundError:
    JST = timezone(timedelta(hours=9))


class EpisodeWriter:
    """Append-only JSONL writer for external teacher / evaluator episodes."""

    def __init__(self, runtime_dir: str | Path = "runtime", *, rotate_latest: bool = True) -> None:
        self.runtime_dir = Path(runtime_dir)
        self.episodes_dir = self.runtime_dir / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.episodes_dir / "latest.jsonl"
        if rotate_latest:
            self._rotate_if_needed(self.latest_path)

    def write(self, payload: Mapping[str, Any]) -> None:
        serializable = dict(payload)
        serializable.setdefault("timestamp_jst", datetime.now(JST).isoformat())
        with self.latest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(serializable, ensure_ascii=False) + "\n")

    def _rotate_if_needed(self, latest_path: Path) -> None:
        if not latest_path.exists() or latest_path.stat().st_size == 0:
            return
        stamp = datetime.now(JST).strftime("%Y%m%d%H%M%S")
        archive_path = latest_path.with_name(f"{stamp}{latest_path.suffix}")
        counter = 1
        while archive_path.exists():
            archive_path = latest_path.with_name(f"{stamp}_{counter}{latest_path.suffix}")
            counter += 1
        shutil.move(str(latest_path), str(archive_path))
