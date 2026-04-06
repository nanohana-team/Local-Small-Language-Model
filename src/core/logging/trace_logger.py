from __future__ import annotations

import json
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:
    ZoneInfo = None
    ZoneInfoNotFoundError = Exception


def _build_jst():
    if ZoneInfo is not None:
        try:
            return ZoneInfo("Asia/Tokyo")
        except ZoneInfoNotFoundError:
            pass
    return timezone(timedelta(hours=9), name="JST")


JST = _build_jst()


def now_jst_iso() -> str:
    return datetime.now(JST).isoformat()


def now_jst_iso() -> str:
    return datetime.now(JST).isoformat(timespec="seconds")


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]

    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=JST)
        return value.isoformat(timespec="seconds")

    return value


class JsonlTraceLogger:
    """
    LSLM v3 用 JSONL トレースロガー。

    使い方:
    - append_trace(trace) で TraceLog / dataclass / dict をそのまま 1 行 JSON に保存
    - append_event(...) でイベント型ログも追記可能
    - append_episode_trace(trace) で RL 用エピソード行として追記可能
    """

    def __init__(
        self,
        log_dir: str | Path = "runtime/traces",
        latest_name: str = "latest_trace.jsonl",
        rotate_on_start: bool = False,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.latest_path = self.log_dir / latest_name

        if rotate_on_start:
            self.rotate_latest()

    def rotate_latest(self) -> Path | None:
        if not self.latest_path.exists() or self.latest_path.stat().st_size <= 0:
            return None

        stamp = datetime.now(JST).strftime("%Y%m%d%H%M%S")
        archive_path = self.log_dir / f"{stamp}_trace.jsonl"
        suffix = 1
        while archive_path.exists():
            archive_path = self.log_dir / f"{stamp}_{suffix}_trace.jsonl"
            suffix += 1

        self.latest_path.rename(archive_path)
        return archive_path

    def new_session_id(self) -> str:
        return f"session_{uuid.uuid4().hex[:12]}"

    def new_turn_id(self) -> str:
        return f"turn_{uuid.uuid4().hex[:12]}"

    def new_episode_id(self) -> str:
        return f"episode_{uuid.uuid4().hex[:12]}"

    def append_record(self, record: Mapping[str, Any]) -> Path:
        jsonable = _to_jsonable(record)
        with self.latest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(jsonable, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")
        return self.latest_path

    def append_event(
        self,
        event: str,
        payload: Mapping[str, Any] | None = None,
        *,
        session_id: str = "",
        turn_id: str = "",
        episode_id: str = "",
    ) -> Path:
        record: dict[str, Any] = {
            "record_type": "event",
            "event": event,
            "timestamp": now_jst_iso(),
        }
        if session_id:
            record["session_id"] = session_id
        if turn_id:
            record["turn_id"] = turn_id
        if episode_id:
            record["episode_id"] = episode_id
        if payload:
            record.update(_to_jsonable(payload))
        return self.append_record(record)

    def start_session(self, extra: Mapping[str, Any] | None = None) -> str:
        session_id = self.new_session_id()
        payload = {"status": "started"}
        if extra:
            payload.update(_to_jsonable(extra))
        self.append_event("session_start", payload, session_id=session_id)
        return session_id

    def end_session(self, session_id: str, extra: Mapping[str, Any] | None = None) -> Path:
        payload = {"status": "ended"}
        if extra:
            payload.update(_to_jsonable(extra))
        return self.append_event("session_end", payload, session_id=session_id)

    def append_trace(self, trace: Any) -> Path:
        jsonable = _to_jsonable(trace)
        if isinstance(jsonable, dict):
            jsonable.setdefault("record_type", "episode_trace")
            jsonable.setdefault("record_version", "rl_trace_v2")
            jsonable.setdefault("timestamp", now_jst_iso())
            jsonable.setdefault("episode_id", self.new_episode_id())
            return self.append_record(jsonable)

        return self.append_record(
            {
                "record_type": "episode_trace",
                "record_version": "rl_trace_v2",
                "episode_id": self.new_episode_id(),
                "timestamp": now_jst_iso(),
                "trace": jsonable,
            }
        )

    def append_episode_trace(self, trace: Any) -> Path:
        return self.append_trace(trace)

    def append_reward(
        self,
        *,
        session_id: str,
        turn_id: str,
        episode_id: str,
        reward: Mapping[str, Any],
    ) -> Path:
        return self.append_event(
            "episode_reward",
            payload={"reward": _to_jsonable(reward)},
            session_id=session_id,
            turn_id=turn_id,
            episode_id=episode_id,
        )

    def append_error(
        self,
        *,
        session_id: str,
        turn_id: str,
        error_type: str,
        message: str,
        detail: Mapping[str, Any] | None = None,
        episode_id: str = "",
    ) -> Path:
        payload: dict[str, Any] = {
            "error_type": error_type,
            "message": message,
        }
        if detail:
            payload["detail"] = _to_jsonable(detail)

        return self.append_event(
            "error",
            payload=payload,
            session_id=session_id,
            turn_id=turn_id,
            episode_id=episode_id,
        )