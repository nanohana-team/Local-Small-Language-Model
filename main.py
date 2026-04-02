from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Sequence


@dataclass
class NodeConfig:
    name: str
    model: str
    listen_port: int
    next_ports: List[int]
    quantization: str = "8bit"


DEFAULT_NODES: List[NodeConfig] = [
    NodeConfig(
        name="A",
        model="microsoft/Phi-3.5-mini-instruct",
        listen_port=3000,
        next_ports=[3001],
        quantization="8bit",
    ),
    NodeConfig(
        name="B",
        model="LiquidAI/LFM2.5-1.2B-Instruct",
        listen_port=3001,
        next_ports=[3002],
        quantization="8bit",
    ),
    NodeConfig(
        name="C",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        listen_port=3002,
        next_ports=[3000],
        quantization="8bit",
    ),
]


class StackRunner:
    def __init__(self) -> None:
        self.processes: List[subprocess.Popen] = []
        self.log_handles: List[IO[str]] = []

    def add(self, proc: subprocess.Popen, log_handle: IO[str]) -> None:
        self.processes.append(proc)
        self.log_handles.append(log_handle)

    def stop_all(self) -> None:
        for proc in reversed(self.processes):
            terminate_process_tree(proc)
        for fh in self.log_handles:
            try:
                fh.close()
            except Exception:
                pass
        self.processes.clear()
        self.log_handles.clear()


RUNNER = StackRunner()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the local LLM ring, inject one prompt, wait for the loop to finish, and clean up."
    )
    parser.add_argument(
        "text",
        nargs="?",
        default="こんにちは。今日の話題を決めてください。",
        help="Initial input text to inject into the loop.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable to use.")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"], help="Inference device.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens per node reply.")
    parser.add_argument("--loops", type=int, default=5, help="Logical loops. Actual TTL is loops × node_count.")
    parser.add_argument("--startup-timeout", type=float, default=180.0, help="Seconds to wait for all ports.")
    parser.add_argument("--poll-interval", type=float, default=0.5, help="Polling interval in seconds.")
    parser.add_argument("--session-timeout", type=float, default=180.0, help="Seconds to wait for a session file after injection.")
    parser.add_argument("--project-root", default=".", help="Project root. Defaults to current directory.")
    parser.add_argument("--keep-running", action="store_true", help="Keep servers alive after the loop finishes.")
    return parser.parse_args()


def resolve_project_root(start: str) -> Path:
    candidate = Path(start).resolve()
    if candidate.is_file():
        candidate = candidate.parent

    search_roots = [candidate, Path.cwd().resolve()]
    for root in search_roots:
        for p in [root, *root.parents]:
            if (p / "config").exists() or (p / "llm.py").exists() or (p / "src").exists():
                return p
    return candidate


def resolve_llm_script(project_root: Path) -> Path:
    candidates = [
        project_root / "llm.py",
        project_root / "llm" / "llm.py",
        project_root / "src" / "chat" / "llm.py",
        project_root / "src" / "chat" / "llm" / "llm.py",
        project_root / "src" / "llm.py",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "llm.py not found. Looked for: " + ", ".join(str(x) for x in candidates)
    )


def resolve_logs_dir(project_root: Path) -> Path:
    candidates = [
        project_root / "logs",
        project_root / "src" / "chat" / "logs",
        project_root / "src" / "logs",
    ]
    for path in candidates:
        if path.exists():
            path.mkdir(parents=True, exist_ok=True)
            return path
    candidates[0].mkdir(parents=True, exist_ok=True)
    return candidates[0]


def build_node_command(
    python_exe: str,
    llm_script: Path,
    node: NodeConfig,
    device: str,
    max_new_tokens: int,
) -> List[str]:
    return [
        python_exe,
        str(llm_script),
        "--model",
        node.model,
        "--listen-port",
        str(node.listen_port),
        "--next-ports",
        *[str(p) for p in node.next_ports],
        "--max-new-tokens",
        str(max_new_tokens),
        "--device",
        device,
        "--quantization",
        node.quantization,
    ]


def wait_for_ports(host: str, ports: Sequence[int], timeout: float, interval: float) -> None:
    start = time.time()
    while True:
        all_ok = True
        for port in ports:
            try:
                with socket.create_connection((host, port), timeout=1.0):
                    pass
            except Exception:
                all_ok = False
                break
        if all_ok:
            return
        if time.time() - start > timeout:
            raise TimeoutError(f"Timeout waiting for ports: {list(ports)}")
        time.sleep(interval)


def send_loop_message(host: str, port: int, message: Dict[str, Any]) -> str:
    payload = json.dumps(message, ensure_ascii=False).encode("utf-8")
    with socket.create_connection((host, port), timeout=10.0) as sock:
        sock.sendall(payload)
        try:
            resp = sock.recv(4096)
            return resp.decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""


def inject_message(text: str, entry_ports: Sequence[int], loops: int) -> str:
    ttl = loops * len(entry_ports)
    message = {
        "session_id": str(uuid.uuid4()),
        "input": text,
        "ttl": ttl,
        "max_ttl": ttl,
        "history": [],
    }
    last_error: Optional[Exception] = None
    print("[START] loop message")
    print(json.dumps(message, ensure_ascii=False, indent=2))
    for port in entry_ports:
        try:
            print(f"[SEND] -> 127.0.0.1:{port}")
            response = send_loop_message("127.0.0.1", port, message)
            print(f"[RECV] {response or '(no response)'}")
            return message["session_id"]
        except Exception as exc:
            last_error = exc
            print(f"[WARN] failed to send to 127.0.0.1:{port}: {exc}")
    raise RuntimeError(f"all entry ports failed: {last_error}")


def wait_for_session_file(sessions_dir: Path, session_id: str, timeout: float, interval: float) -> Path:
    target = sessions_dir / f"{session_id}.json"
    start = time.time()
    stable_count = 0
    last_size = -1
    while True:
        if target.exists():
            size = target.stat().st_size
            if size > 0 and size == last_size:
                stable_count += 1
                if stable_count >= 3:
                    return target
            else:
                stable_count = 0
                last_size = size
        if time.time() - start > timeout:
            raise TimeoutError(f"Session file was not finalized in time: {target}")
        time.sleep(interval)


def print_session_summary(session_path: Path) -> None:
    try:
        data = json.loads(session_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] failed to read session file: {exc}")
        return

    history = data.get("history", [])
    print("")
    print("=== SESSION SUMMARY ===")
    print(f"[INFO] session_file = {session_path}")
    print(f"[INFO] session_id   = {data.get('session_id', session_path.stem)}")
    print(f"[INFO] turns        = {len(history)}")
    if history:
        last = history[-1]
        print(f"[INFO] last_model   = {last.get('model')}")
        print(f"[INFO] last_text    = {str(last.get('text', ''))[:300]}")


def terminate_process_tree(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass


def main() -> int:
    args = parse_args()
    project_root = resolve_project_root(args.project_root)
    llm_script = resolve_llm_script(project_root)
    logs_dir = resolve_logs_dir(project_root)
    sessions_dir = logs_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    runtime_log_dir = logs_dir / "runtime"
    runtime_log_dir.mkdir(parents=True, exist_ok=True)

    atexit.register(RUNNER.stop_all)

    print("=== START LOCAL LLM LOOP ===")
    print(f"[INFO] project_root    : {project_root}")
    print(f"[INFO] llm_script      : {llm_script}")
    print(f"[INFO] logs_dir        : {logs_dir}")
    print(f"[INFO] sessions_dir    : {sessions_dir}")
    print(f"[INFO] device          : {args.device}")
    print(f"[INFO] max_new_tokens  : {args.max_new_tokens}")
    print(f"[INFO] loops           : {args.loops}")

    entry_ports = [node.listen_port for node in DEFAULT_NODES]

    try:
        for node in DEFAULT_NODES:
            cmd = build_node_command(
                python_exe=args.python,
                llm_script=llm_script,
                node=node,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
            )
            log_path = runtime_log_dir / f"node_{node.listen_port}.log"
            log_fh = open(log_path, "w", encoding="utf-8")
            print(f"[LAUNCH] {node.name} -> {log_path}")
            print(f"[CMD] {' '.join(cmd)}")
            proc = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )
            RUNNER.add(proc, log_fh)
            time.sleep(1.0)

        print(f"[WAIT] waiting for ports: {entry_ports}")
        wait_for_ports("127.0.0.1", entry_ports, args.startup_timeout, args.poll_interval)
        print("[WAIT] all ports ready")

        session_id = inject_message(args.text, entry_ports, args.loops)
        session_path = wait_for_session_file(sessions_dir, session_id, args.session_timeout, args.poll_interval)
        print_session_summary(session_path)

        if args.keep_running:
            print("[INFO] keep-running enabled. Press Ctrl+C to stop.")
            while True:
                time.sleep(1.0)

        return 0
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C received. Stopping...")
        return 130
    finally:
        if not args.keep_running:
            RUNNER.stop_all()
            print("[INFO] shutdown complete")


if __name__ == "__main__":
    raise SystemExit(main())
