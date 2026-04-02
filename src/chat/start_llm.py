import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List

def wait_for_ports(host: str, ports: List[int], timeout: float = 30.0, interval: float = 0.5) -> None:
    start = time.time()
    while True:
        if all(_can_connect(host, p) for p in ports):
            return
        if time.time() - start > timeout:
            raise TimeoutError(f"Timeout waiting for ports: {ports}")
        time.sleep(interval)

def _can_connect(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except Exception:
        return False

def launch_in_new_console(cmd: List[str], cwd: Path, log_file: Path) -> subprocess.Popen:
    if os.name == "nt":
        quoted = subprocess.list2cmdline(cmd)
        return subprocess.Popen(f'cmd.exe /k {quoted}', cwd=str(cwd), creationflags=subprocess.CREATE_NEW_CONSOLE)
    with log_file.open("a", encoding="utf-8") as fh:
        return subprocess.Popen(cmd, cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT)

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

def terminate_process_tree(proc: subprocess.Popen | None, name: str) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception as exc:
        print(f"[WARN] Failed to terminate {name}: {exc}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--startup-delay", type=float, default=1.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--loop-text", default="雑談してください。")
    parser.add_argument("--loops", type=int, default=5)
    parser.add_argument("--entry-ports", type=int, nargs="+", default=[3000,3001,3002])
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    chat_dir = Path(__file__).resolve().parent
    logs_dir = chat_dir / "logs" / "launcher" / time.strftime("%Y%m%d_%H%M%S")
    logs_dir.mkdir(parents=True, exist_ok=True)
    llm_proc = combine_proc = loop_proc = None
    try:
        llm_proc = launch_in_new_console([args.python, str(chat_dir / 'start_llm_server.py'), '--device', args.device, '--max-new-tokens', str(args.max_new_tokens)], chat_dir, logs_dir / 'start_llm_server.log')
        time.sleep(args.startup_delay)
        combine_proc = launch_in_new_console([args.python, str(chat_dir / 'combine_logs.py'), '--poll-interval', str(args.poll_interval)], chat_dir, logs_dir / 'combine_logs.log')
        wait_for_ports('127.0.0.1', args.entry_ports)
        time.sleep(args.startup_delay)
        loop_proc = launch_in_new_console([args.python, str(chat_dir / 'loop.py'), args.loop_text, '--entry-ports', *[str(p) for p in args.entry_ports], '--loops', str(args.loops)], chat_dir, logs_dir / 'loop.log')
        (logs_dir / 'launch_manifest.json').write_text(json.dumps({'created_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'device': args.device, 'entry_ports': args.entry_ports, 'loop_text': args.loop_text, 'loops': args.loops, 'start_llm_server_pid': llm_proc.pid if llm_proc else None, 'combine_logs_pid': combine_proc.pid if combine_proc else None, 'loop_pid': loop_proc.pid if loop_proc else None}, ensure_ascii=False, indent=2), encoding='utf-8')
        while True:
            time.sleep(1.0)
            if combine_proc and combine_proc.poll() is not None:
                break
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C received. Stopping all processes...")
    finally:
        terminate_process_tree(loop_proc, 'loop.py')
        terminate_process_tree(combine_proc, 'combine_logs.py')
        terminate_process_tree(llm_proc, 'start_llm_server.py')
        for port in args.entry_ports:
            kill_process_by_port(port)

if __name__ == '__main__':
    main()
