from __future__ import annotations

import atexit
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TextIO


_TS_KEY_CANDIDATES = (
    "session_end_timestamp",
    "timestamp",
    "ended_at",
    "ended_at_iso",
)

_LOG_HANDLE: TextIO | None = None
_ORIGINAL_STDOUT: TextIO | None = None
_ORIGINAL_STDERR: TextIO | None = None


class TeeStream:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams
        self._buffer = ""

    def write(self, data: str) -> int:
        self._buffer += data

        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)

            if line.strip():
                ts = _now_local_text()
                line = f"[{ts}] {line}"

            line = line + "\n"

            for s in self.streams:
                try:
                    s.write(line)
                except Exception:
                    pass

        return len(data)

    def flush(self) -> None:
        if self._buffer:
            line = self._buffer
            self._buffer = ""

            if line.strip():
                ts = _now_local_text()
                line = f"[{ts}] {line}"

            for s in self.streams:
                try:
                    s.write(line)
                except Exception:
                    pass

        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        for s in self.streams:
            try:
                if s.isatty():
                    return True
            except Exception:
                pass
        return False


def load_dotenv_file(dotenv_path: str | Path = ".env", override: bool = False) -> None:
    path = Path(dotenv_path)
    if not path.exists():
        return

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8-sig")

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        if override or key not in os.environ:
            os.environ[key] = value


def _now_local_dt() -> datetime:
    return datetime.now().astimezone()


def _now_local_iso() -> str:
    return _now_local_dt().isoformat(timespec="seconds")


def _now_local_text() -> str:
    return _now_local_dt().strftime("%Y-%m-%d %H:%M:%S")


def _safe_read_lines(path: Path) -> list[str]:
    encodings = [
        "utf-8",
        "utf-8-sig",
        "cp932",
        "mbcs",
    ]

    for enc in encodings:
        try:
            return path.read_text(encoding=enc).splitlines()
        except Exception:
            pass

    try:
        return path.read_bytes().decode("utf-8", errors="replace").splitlines()
    except Exception:
        return []


def _strip_log_prefix(line: str) -> str:
    line = line.strip()
    if not line.startswith("["):
        return line

    close_idx = line.find("] ")
    if close_idx == -1:
        return line

    prefix = line[1:close_idx]
    if len(prefix) == 19 and prefix[4] == "-" and prefix[7] == "-" and prefix[10] == " ":
        return line[close_idx + 2 :].strip()

    return line


def _extract_timestamp_from_latest_log(path: Path) -> str | None:
    lines = _safe_read_lines(path)

    for raw in reversed(lines):
        line = _strip_log_prefix(raw)
        if not line:
            continue

        try:
            obj = json.loads(line)
        except Exception:
            continue

        if not isinstance(obj, dict):
            continue

        for key in _TS_KEY_CANDIDATES:
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return None


def _timestamp_to_archive_stem(timestamp_text: str | None) -> str:
    if timestamp_text:
        text = timestamp_text.strip()

        try:
            if "T" in text:
                normalized = text.replace("Z", "+00:00")
                dt = datetime.fromisoformat(normalized)
                dt = dt.astimezone()
                return dt.strftime("%Y%m%d%H%M%S")
        except Exception:
            pass

        try:
            dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S").astimezone()
            return dt.strftime("%Y%m%d%H%M%S")
        except Exception:
            pass

    return _now_local_dt().strftime("%Y%m%d%H%M%S")


def rotate_previous_latest_logs(log_dir: str | Path) -> None:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    latest_path = log_dir / "latest.log"
    if not latest_path.exists():
        return

    stamp_text = _extract_timestamp_from_latest_log(latest_path)
    stem14 = _timestamp_to_archive_stem(stamp_text)

    archive_path = log_dir / f"{stem14}.log"
    index = 0
    while archive_path.exists():
        archive_path = log_dir / f"{stem14}_{index}.log"
        index += 1

    latest_path.rename(archive_path)
    print(f"[LOG ROTATE] archived latest.log -> {archive_path.name}", flush=True)


def _write_session_end_marker(log_dir: str | Path) -> None:
    log_dir = Path(log_dir)
    latest_path = log_dir / "latest.log"

    ended_at = _now_local_text()
    ended_at_unix = time.time()

    try:
        with latest_path.open("a", encoding="utf-8", errors="replace") as f:
            f.write(
                json.dumps(
                    {
                        "event": "session_end",
                        "session_end_timestamp": ended_at,
                        "session_end_unix": ended_at_unix,
                        "source_file": "latest.log",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()
    except Exception as e:
        try:
            print(f"[LOG ROTATE][WARN] failed to append session_end: {e}", flush=True)
        except Exception:
            pass


def _close_log_streams() -> None:
    global _LOG_HANDLE, _ORIGINAL_STDOUT, _ORIGINAL_STDERR

    try:
        if _ORIGINAL_STDOUT is not None:
            sys.stdout = _ORIGINAL_STDOUT
        if _ORIGINAL_STDERR is not None:
            sys.stderr = _ORIGINAL_STDERR
    except Exception:
        pass

    try:
        if _LOG_HANDLE is not None:
            _LOG_HANDLE.flush()
            _LOG_HANDLE.close()
    except Exception:
        pass

    _LOG_HANDLE = None
    _ORIGINAL_STDOUT = None
    _ORIGINAL_STDERR = None


def finalize_latest_log(log_dir: str | Path) -> None:
    _write_session_end_marker(log_dir)
    _close_log_streams()


def register_log_finalizer(log_dir: str | Path) -> None:
    atexit.register(finalize_latest_log, Path(log_dir))


def setup_latest_log_capture(log_dir: str | Path) -> Path:
    global _LOG_HANDLE, _ORIGINAL_STDOUT, _ORIGINAL_STDERR

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    latest_path = log_dir / "latest.log"
    _LOG_HANDLE = latest_path.open("a", encoding="utf-8", buffering=1, errors="replace")

    _ORIGINAL_STDOUT = sys.stdout
    _ORIGINAL_STDERR = sys.stderr

    sys.stdout = TeeStream(_ORIGINAL_STDOUT, _LOG_HANDLE)
    sys.stderr = TeeStream(_ORIGINAL_STDERR, _LOG_HANDLE)

    print(
        json.dumps(
            {
                "event": "session_start",
                "session_start_timestamp": _now_local_text(),
                "source_file": "latest.log",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    return latest_path


def get_latest_log_path(log_dir: str | Path) -> Path:
    return Path(log_dir) / "latest.log"


def prepare_log_session(log_dir: str | Path) -> Path:
    rotate_previous_latest_logs(log_dir)
    latest_path = setup_latest_log_capture(log_dir)
    register_log_finalizer(log_dir)
    return latest_path