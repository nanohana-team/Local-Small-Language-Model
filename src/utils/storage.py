# src/utils/storage.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class StorageManager:
    def __init__(
        self,
        log_dir: Path,
        episode_file: str = "episodes.jsonl",
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.episode_path = self.log_dir / episode_file

    def save_episode(self, episode: Dict[str, Any]) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with self.episode_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(episode, ensure_ascii=False) + "\n")

    def load_all_episodes(self) -> List[Dict[str, Any]]:
        if not self.episode_path.exists():
            return []

        rows: List[Dict[str, Any]] = []
        with self.episode_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        rows.append(row)
                except json.JSONDecodeError:
                    continue
        return rows

    def load_recent_episodes(self, limit: int = 100) -> List[Dict[str, Any]]:
        rows = self.load_all_episodes()
        if limit <= 0:
            return rows
        return rows[-limit:]

    def save_json(self, relative_path: str, obj: Any) -> Path:
        path = self.log_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return path

    def load_json(self, relative_path: str) -> Optional[Any]:
        path = self.log_dir / relative_path
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)