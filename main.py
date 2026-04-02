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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

@dataclass
class NodeConfig:
    name: str
    model: str
    listen_port: int
    next_ports: List[int]
    quantization: str = "8bit"

DEFAULT_NODES: List[NodeConfig] = [
    NodeConfig("A", "microsoft/Phi-3.5-mini-instruct", 3000, [3001], "8bit"),
    NodeConfig("B", "LiquidAI/LFM2.5-1.2B-Instruct", 3001, [3002], "8bit"),
    NodeConfig("C", "Qwen/Qwen2.5-1.5B-Instruct", 3002, [3000], "8bit"),
]

class StackRunner:
    def __init__(self) -> None:
        self.processes: List[subprocess.Popen] = []
    def add(self, proc: subprocess.Popen) -> None:
        self.processes.append(proc)
    def stop_all(self) -> None:
        for proc in reversed(self.processes):
            terminate_process_tree(proc)
        self.processes.clear()

RUNNER = StackRunner()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the local LLM ring in separate windows, inject one prompt, wait for completion, and clean up.")
    parser.add_argument("text", nargs="?", default="こんにちは。今日の話題を決めてください。", help="Initial input text to inject into the loop.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--loops", type=int, default=5)
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--session-timeout", type=float, default=180.0)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--keep-running", action="store_true")
    parser.add_argument("--combine-logs", action="store_true")
    return parser.parse_args()

def resolve_project_root(start: str) -> Path:
    candidate = Path(start).resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for root in [candidate, Path.cwd().resolve()]:
        for p in [root, *root.parents]:
            if (p / "config").exists() or (p / "llm.py").exists() or (p / "src").exists():
                return p
    return candidate

def resolve_script(project_root: Path, names: Sequence[str]) -> Path:
    for name in names:
        path = project_root / name
        if path.exists():
            return path
    raise FileNotFoundError("script not found. looked for: " + ", ".join(str(project_root / x) for x in names))

def resolve_llm_script(project_root: Path) -> Path:
    return resolve_script(project_root, ["llm.py", "llm/llm.py", "src/chat/llm.py", "src/chat/llm/llm.py", "src/llm.py"])

def resolve_logs_dir(project_root: Path) -> Path:
    candidates = [project_root / "logs", project_root / "src" / "chat" / "logs", project_root / "src" / "logs"]
    for path in candidates:
        if path.exists():
            path.mkdir(parents=True, exist_ok=True)
            return path
    candidates[0].mkdir(parents=True, exist_ok=True)
    return candidates[0]

def build_node_command(python_exe: str, llm_script: Path, node: NodeConfig, device: str, max_new_tokens: int) -> List[str]:
    return [python_exe, str(llm_script), "--model", node.model, "--listen-port", str(node.listen_port), "--next-ports", *[str(p) for p in node.next_ports], "--max-new-tokens", str(max_new_tokens), "--device", device, "--quantization", node.quantization]

def wait_for_ports(host: str, ports: Sequence[int], timeout: float, interval: float) -> None:
    start = time.time()
    while True:
        ok = True
        for port in ports:
            try:
                with socket.create_connection((host, port), timeout=1.0):
                    pass
            except Exception:
                ok = False
                break
        if ok:
            return
        if time.time() - start > timeout:
            raise TimeoutError(f"Timeout waiting for ports: {list(ports)}")
        time.sleep(interval)

def send_loop_message(host: str, port: int, message: Dict[str, Any]) -> str:
    payload = json.dumps(message, ensure_ascii=False).encode("utf-8")
    with socket.create_connection((host, port), timeout=10.0) as sock:
        sock.sendall(payload)
        try:
            return sock.recv(4096).decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""

def inject_message(text: str, entry_ports: Sequence[int], loops: int, launcher_log: Path) -> str:
    ttl = loops * len(entry_ports)
    message = {"session_id": str(uuid.uuid4()), "input": text, "ttl": ttl, "max_ttl": ttl, "history": []}
    launcher_log.write_text(json.dumps({"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "message": message}, ensure_ascii=False, indent=2), encoding="utf-8")
    last_error: Optional[Exception] = None
    print(json.dumps(message, ensure_ascii=False, indent=2))
    for port in entry_ports:
        try:
            response = send_loop_message("127.0.0.1", port, message)
            print(f"[SEND] 127.0.0.1:{port} -> {response or '(no response)'}")
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
        history = data.get("history", [])
        print("\n=== SESSION SUMMARY ===")
        print(f"[INFO] session_file : {session_path}")
        print(f"[INFO] session_id   : {data.get('session_id')}")
        print(f"[INFO] model        : {data.get('model')}")
        print(f"[INFO] updated_at   : {data.get('updated_at')}")
        print(f"[INFO] history_items: {len(history)}")
        if history:
            print("[INFO] last_reply   :", history[-1].get("text", ""))
    except Exception as exc:
        print(f"[WARN] failed to summarize session file: {exc}")

def kill_process_by_port(port: int) -> None:
    if os.name != "nt":
        return
    try:
        result = subprocess.check_output(f'netstat -ano | findstr :{port}', shell=True, encoding='utf-8')
        for line in result.strip().splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[-1].isdigit():
                subprocess.run(["taskkill", "/PID", parts[-1], "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        pass

def terminate_process_tree(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        pass

def launch_in_new_console(cmd: List[str], cwd: Path, log_file: Path) -> subprocess.Popen:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        quoted = subprocess.list2cmdline(cmd)
        full_cmd = f'{quoted} 1>> "{log_file}" 2>&1'
        return subprocess.Popen(["cmd.exe", "/k", full_cmd], cwd=str(cwd), creationflags=subprocess.CREATE_NEW_CONSOLE)
    with log_file.open("a", encoding="utf-8") as fh:
        return subprocess.Popen(cmd, cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT)

def main() -> int:
    args = parse_args()
    project_root = resolve_project_root(args.project_root)
    llm_script = resolve_llm_script(project_root)
    logs_dir = resolve_logs_dir(project_root)
    sessions_dir = logs_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    launcher_dir = logs_dir / "launcher" / run_id
    launcher_dir.mkdir(parents=True, exist_ok=True)
    (launcher_dir / "run_manifest.json").write_text(json.dumps({"run_id": run_id, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "project_root": str(project_root), "llm_script": str(llm_script), "device": args.device, "max_new_tokens": args.max_new_tokens, "loops": args.loops, "nodes": [asdict(node) for node in DEFAULT_NODES], "combine_logs": args.combine_logs}, ensure_ascii=False, indent=2), encoding="utf-8")
    atexit.register(RUNNER.stop_all)
    print(f"[INFO] launcher_dir : {launcher_dir}")
    if args.combine_logs:
        combine_script = resolve_script(project_root, ["combine_logs.py"])
        proc = launch_in_new_console([args.python, str(combine_script)], project_root, launcher_dir / "combine_logs.log")
        RUNNER.add(proc)
    for node in DEFAULT_NODES:
        cmd = build_node_command(args.python, llm_script, node, args.device, args.max_new_tokens)
        proc = launch_in_new_console(cmd, project_root, launcher_dir / f"server_{node.name}_port_{node.listen_port}.log")
        RUNNER.add(proc)
        time.sleep(1.0)
    entry_ports = [node.listen_port for node in DEFAULT_NODES]
    wait_for_ports("127.0.0.1", entry_ports, args.startup_timeout, args.poll_interval)
    session_id = inject_message(args.text, entry_ports, args.loops, launcher_dir / "injected_message.json")
    session_path = wait_for_session_file(sessions_dir, session_id, args.session_timeout, args.poll_interval)
    print_session_summary(session_path)
    (launcher_dir / "final_session_path.txt").write_text(str(session_path), encoding="utf-8")
    if args.keep_running:
        print("[INFO] keep-running enabled. Servers stay alive.")
        return 0
    RUNNER.stop_all()
    for port in entry_ports:
        kill_process_by_port(port)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
