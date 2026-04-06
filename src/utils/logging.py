from __future__ import annotations

import atexit
import ctypes
import logging
import os
import re
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo


JST = ZoneInfo("Asia/Tokyo")

# 例:
# 2026-04-06 12:13:33 [INFO] test info
_LOG_LINE_TS_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")

# ANSI color
_RESET = "\033[0m"
_GRAY = "\033[90m"
_WHITE = "\033[97m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_WHITE_ON_RED = "\033[41;97m"


@dataclass(slots=True)
class RotationResult:
    latest_rotated_to: Path | None = None
    debug_rotated_to: Path | None = None
    chosen_stamp: str = ""


class JSTFormatter(logging.Formatter):
    """
    JST固定、秒精度、形式:
    2026-04-06 12:13:33 [INFO] message
    """

    def __init__(self) -> None:
        super().__init__(fmt="%(asctime)s [%(levelname)s] %(message)s")

    def formatTime(
        self,
        record: logging.LogRecord,
        datefmt: str | None = None,
    ) -> str:
        dt = datetime.fromtimestamp(record.created, tz=JST)
        return dt.strftime("%Y-%m-%d %H:%M:%S")


class ColorConsoleFormatter(JSTFormatter):
    """
    コンソール専用カラー。
    DEBUG    = グレー
    INFO     = 白
    WARNING  = 黄色
    ERROR    = 赤
    CRITICAL = 赤背景 + 白文字
    """

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)

        if record.levelno >= logging.CRITICAL:
            return f"{_WHITE_ON_RED}{text}{_RESET}"

        if record.levelno >= logging.ERROR:
            return f"{_RED}{text}{_RESET}"

        if record.levelno >= logging.WARNING:
            return f"{_YELLOW}{text}{_RESET}"

        if record.levelno >= logging.INFO:
            return f"{_WHITE}{text}{_RESET}"

        return f"{_GRAY}{text}{_RESET}"


def _enable_windows_vt_mode() -> None:
    """
    Windows コンソールで ANSI エスケープを有効化。
    失敗してもログ機能自体は継続。
    """
    if os.name != "nt":
        return

    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        if handle == 0 or handle == -1:
            return

        mode = ctypes.c_uint()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) == 0:
            return

        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        new_mode = mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING
        kernel32.SetConsoleMode(handle, new_mode)
    except Exception:
        pass


def now_jst() -> datetime:
    return datetime.now(JST)


def now_jst_text() -> str:
    return now_jst().strftime("%Y-%m-%d %H:%M:%S")


def _to_filename_stamp(dt: datetime) -> str:
    return dt.astimezone(JST).strftime("%Y%m%d%H%M%S")


def _tail_text(path: Path, max_bytes: int = 128 * 1024) -> str:
    if not path.exists() or path.stat().st_size <= 0:
        return ""

    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        read_size = min(size, max_bytes)
        f.seek(-read_size, os.SEEK_END)
        data = f.read()

    return data.decode("utf-8", errors="replace")


def _extract_last_timestamp(path: Path) -> datetime | None:
    """
    ファイル末尾側から最後のログ時刻を拾う。
    形式:
    2026-04-06 12:13:33 [INFO] ...
    """
    text = _tail_text(path)
    if text:
        for line in reversed(text.splitlines()):
            match = _LOG_LINE_TS_RE.match(line.strip())
            if not match:
                continue
            raw = match.group("ts")
            try:
                return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S").replace(tzinfo=JST)
            except ValueError:
                continue

    if path.exists():
        return datetime.fromtimestamp(path.stat().st_mtime, tz=JST)

    return None


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    parent = path.parent
    stem = path.stem
    suffix = path.suffix

    index = 1
    while True:
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def _safe_rename(src: Path, dst: Path) -> Path | None:
    if not src.exists():
        return None

    if src.stat().st_size == 0:
        src.unlink(missing_ok=True)
        return None

    dst = _unique_path(dst)
    src.rename(dst)
    return dst


def rotate_previous_logs(
    log_dir: str | Path = "logs",
    latest_name: str = "latest.log",
    debug_name: str = "debug.log",
) -> RotationResult:
    """
    起動時に前回ログを回転する。

    例:
    logs/latest.log -> logs/20260406121334.log
    logs/debug.log  -> logs/20260406121334_debug.log
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    latest_path = log_dir / latest_name
    debug_path = log_dir / debug_name

    latest_ts = _extract_last_timestamp(latest_path)
    debug_ts = _extract_last_timestamp(debug_path)

    timestamps = [ts for ts in (latest_ts, debug_ts) if ts is not None]
    if not timestamps:
        return RotationResult()

    chosen_dt = max(timestamps)
    stamp = _to_filename_stamp(chosen_dt)

    latest_rotated_to = _safe_rename(latest_path, log_dir / f"{stamp}.log")
    debug_rotated_to = _safe_rename(debug_path, log_dir / f"{stamp}_debug.log")

    return RotationResult(
        latest_rotated_to=latest_rotated_to,
        debug_rotated_to=debug_rotated_to,
        chosen_stamp=stamp,
    )


def _clear_root_handlers() -> None:
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        try:
            handler.flush()
            handler.close()
        except Exception:
            pass


def _install_exception_hooks() -> None:
    def handle_exception(exc_type, exc_value, exc_traceback) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logging.getLogger("uncaught").exception(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    def handle_thread_exception(args: threading.ExceptHookArgs) -> None:
        logging.getLogger("uncaught").exception(
            "Uncaught thread exception in %s",
            getattr(args.thread, "name", "<unknown-thread>"),
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = handle_exception
    threading.excepthook = handle_thread_exception
    logging.captureWarnings(True)


def _register_atexit(app_name: str) -> None:
    def _on_exit() -> None:
        try:
            logging.getLogger(app_name).info("process_exit")
        except Exception:
            pass

    atexit.register(_on_exit)


def setup_logging(
    app_name: str = "app",
    log_dir: str | Path = "logs",
    latest_name: str = "latest.log",
    debug_name: str = "debug.log",
    console_level: int = logging.INFO,
) -> RotationResult:
    """
    構成:
    - logs/debug.log : DEBUG / INFO / WARNING / ERROR / CRITICAL
    - logs/latest.log: INFO / WARNING / ERROR / CRITICAL
    - console        : INFO / WARNING / ERROR / CRITICAL

    出力形式:
    2026-04-06 12:13:33 [DEBUG] test debug
    """
    rotation = rotate_previous_logs(
        log_dir=log_dir,
        latest_name=latest_name,
        debug_name=debug_name,
    )

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    latest_path = log_dir / latest_name
    debug_path = log_dir / debug_name

    _clear_root_handlers()
    _enable_windows_vt_mode()

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    file_formatter = JSTFormatter()
    console_formatter = ColorConsoleFormatter()

    # debug.log: DEBUG以上すべて
    debug_handler = logging.FileHandler(debug_path, mode="a", encoding="utf-8")
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(file_formatter)

    # latest.log: INFO以上（DEBUGは入れない）
    latest_handler = logging.FileHandler(latest_path, mode="a", encoding="utf-8")
    latest_handler.setLevel(logging.INFO)
    latest_handler.setFormatter(file_formatter)

    # console: INFO以上、色付き
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)

    root.addHandler(debug_handler)
    root.addHandler(latest_handler)
    root.addHandler(console_handler)

    _install_exception_hooks()
    _register_atexit(app_name)

    logger = logging.getLogger(app_name)
    logger.info("logging_started")

    if rotation.chosen_stamp:
        logger.info(
            "rotated_previous_logs stamp=%s latest=%s debug=%s",
            rotation.chosen_stamp,
            str(rotation.latest_rotated_to) if rotation.latest_rotated_to else "-",
            str(rotation.debug_rotated_to) if rotation.debug_rotated_to else "-",
        )

    return rotation


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name)


def shutdown_logging() -> None:
    logging.getLogger("app").info("logging_shutdown")
    logging.shutdown()