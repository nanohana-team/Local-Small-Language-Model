import argparse
import json
import time
from pathlib import Path
from typing import Any


class CombineLogs:
    def __init__(
        self,
        logs_dir: Path,
        output_path: Path,
        poll_interval: float = 1.0,
    ) -> None:
        self.logs_dir = logs_dir
        self.output_path = output_path
        self.poll_interval = poll_interval

        self.file_offsets: dict[Path, int] = {}
        self.sessions: dict[str, dict[str, Any]] = {}
        self.seen_events: set[str] = set()

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def ensure_session(self, session_id: str) -> dict[str, Any]:
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "session_id": session_id,
                "input": "",
                "last_ttl": None,
                "history": [],
                "status": "running",
                "updated_at": None,
                "last_event": None,
                "error": None,
                "events": [],
            }
        return self.sessions[session_id]

    def build_event_id(self, source_file: Path, byte_offset: int, event: dict[str, Any]) -> str:
        session_id = event.get("session_id", "no-session")
        timestamp = event.get("timestamp", "")
        event_name = event.get("event", "")
        port = event.get("port", "")
        return "|".join(
            [source_file.name, str(byte_offset), str(session_id), str(timestamp), str(event_name), str(port)]
        )

    def update_session(self, source_file: Path, event: dict[str, Any]) -> None:
        session_id = event.get("session_id")
        if not session_id:
            return

        session = self.ensure_session(session_id)
        session["updated_at"] = event.get("timestamp")
        session["last_event"] = event.get("event")

        if event.get("input"):
            session["input"] = event["input"]

        if event.get("ttl") is not None:
            session["last_ttl"] = event.get("ttl")

        if isinstance(event.get("history"), list):
            session["history"] = event["history"]

        event_name = event.get("event")
        if event_name == "final":
            session["status"] = "final"
        elif event_name == "end":
            session["status"] = "end"
        elif event_name == "error":
            session["status"] = "error"
            session["error"] = event.get("error")
        elif session["status"] != "error":
            session["status"] = "running"

        session["events"].append({"source_file": source_file.name, **event})

    def write_output(self) -> None:
        ordered = dict(sorted(self.sessions.items(), key=lambda x: x[0]))
        with self.output_path.open("w", encoding="utf-8") as f:
            json.dump(ordered, f, ensure_ascii=False, indent=2)
            f.flush()

    def process_file(self, path: Path) -> bool:
        changed = False
        current_offset = self.file_offsets.get(path, 0)

        with path.open("r", encoding="utf-8") as f:
            f.seek(current_offset)
            while True:
                byte_offset = f.tell()
                line = f.readline()
                if not line:
                    break

                raw_line = line.strip()
                if not raw_line:
                    continue

                try:
                    event = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                event_id = self.build_event_id(path, byte_offset, event)
                if event_id in self.seen_events:
                    continue

                self.seen_events.add(event_id)
                self.update_session(path, event)
                changed = True

            self.file_offsets[path] = f.tell()

        return changed

    def scan_once(self) -> bool:
        changed = False
        files = sorted(self.logs_dir.glob("*.jsonl"))

        for path in files:
            if self.process_file(path):
                changed = True

        if changed:
            self.write_output()

        return changed

    def run(self) -> None:
        print(f"[START] watching logs: {self.logs_dir}")
        print(f"[OUT  ] merged json : {self.output_path}")

        while True:
            try:
                changed = self.scan_once()
                if changed:
                    print(f"[SYNC ] sessions={len(self.sessions)} updated")
                time.sleep(self.poll_interval)
            except KeyboardInterrupt:
                print("\n[STOP ] combine_logs stopped")
                break
            except Exception as exc:
                print(f"[ERROR] {exc}")
                time.sleep(self.poll_interval)


def default_paths() -> tuple[Path, Path]:
    base_dir = Path(__file__).resolve().parent
    candidates = [
        (base_dir / "logs", base_dir / "logs" / "sessions_merged.json"),
        (base_dir / "llm" / "logs", base_dir / "logs" / "sessions_merged.json"),
    ]
    for logs_dir, output_path in candidates:
        if logs_dir.exists():
            return logs_dir, output_path
    return candidates[0]


def parse_args() -> argparse.Namespace:
    default_logs_dir, default_output_path = default_paths()

    parser = argparse.ArgumentParser(
        description="Combine logs/*.jsonl in real time and write to a single merged JSON file."
    )
    parser.add_argument("--logs-dir", type=Path, default=default_logs_dir)
    parser.add_argument("--output", type=Path, default=default_output_path)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    combiner = CombineLogs(
        logs_dir=args.logs_dir,
        output_path=args.output,
        poll_interval=args.poll_interval,
    )
    combiner.run()


if __name__ == "__main__":
    main()
