import argparse
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

    print(f"[WAIT] Waiting for ports: {ports}")

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
            print("[WAIT] All ports ready")
            return

        if time.time() - start > timeout:
            raise TimeoutError(f"Timeout waiting for ports: {ports}")

        time.sleep(interval)


def launch_in_new_console(cmd: List[str], cwd: Path) -> subprocess.Popen:
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_CONSOLE

    return subprocess.Popen(
        cmd,
        cwd=str(cwd),
        creationflags=creationflags,
    )
def kill_process_by_port(port: int) -> None:
    try:
        result = subprocess.check_output(
            f'netstat -ano | findstr :{port}',
            shell=True,
            encoding="utf-8"
        )

        lines = result.strip().splitlines()

        pids = set()
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                pid = parts[-1]
                if pid.isdigit():
                    pids.add(pid)

        for pid in pids:
            print(f"[STOP] Killing PID {pid} (port {port})")
            subprocess.run(
                ["taskkill", "/PID", pid, "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    except subprocess.CalledProcessError:
        # ポート使ってるプロセスなし
        pass

def terminate_process_tree(proc: subprocess.Popen | None, name: str) -> None:
    if proc is None:
        return

    if proc.poll() is not None:
        print(f"[STOP] {name} already exited")
        return

    print(f"[STOP] Terminating {name} (pid={proc.pid})")

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
    except Exception as exc:
        print(f"[WARN] Failed to terminate {name}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch LLM servers, log combiner, and loop starter in separate consoles."
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--startup-delay", type=float, default=1.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--loop-text", default="雑談してください。")
    parser.add_argument("--loops", type=int, default=5)
    parser.add_argument("--entry-ports", type=int, nargs="+", default=[3000, 3001, 3002])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    this_file = Path(__file__).resolve()
    chat_dir = this_file.parent

    start_llm_server_script = chat_dir / "start_llm_server.py"
    combine_logs_script = chat_dir / "combine_logs.py"
    loop_script = chat_dir / "loop.py"

    llm_proc = None
    combine_proc = None
    loop_proc = None

    try:
        llm_cmd = [
            args.python,
            str(start_llm_server_script),
            "--device",
            args.device,
            "--max-new-tokens",
            str(args.max_new_tokens),
        ]

        combine_cmd = [
            args.python,
            str(combine_logs_script),
            "--poll-interval",
            str(args.poll_interval),
        ]

        loop_cmd = [
            args.python,
            str(loop_script),
            args.loop_text,
            "--entry-ports",
            *[str(p) for p in args.entry_ports],
            "--loops",
            str(args.loops),
        ]

        print("=== START LLM STACK ===")
        print(f"[INFO] Device      : {args.device}")
        print(f"[INFO] Loop text   : {args.loop_text}")
        print(f"[INFO] Loops       : {args.loops}")
        print(f"[INFO] Entry ports : {args.entry_ports}")

        print(f"[LAUNCH] {' '.join(llm_cmd)}")
        llm_proc = launch_in_new_console(llm_cmd, cwd=chat_dir)

        time.sleep(args.startup_delay)

        print(f"[LAUNCH] {' '.join(combine_cmd)}")
        combine_proc = launch_in_new_console(combine_cmd, cwd=chat_dir)

        wait_for_ports("127.0.0.1", args.entry_ports)

        time.sleep(args.startup_delay)

        print(f"[LAUNCH] {' '.join(loop_cmd)}")
        loop_proc = launch_in_new_console(loop_cmd, cwd=chat_dir)

        print("=== ALL PROCESSES LAUNCHED ===")
        print(f"[PID] start_llm_server.py : {llm_proc.pid}")
        print(f"[PID] combine_logs.py     : {combine_proc.pid}")
        print(f"[PID] loop.py             : {loop_proc.pid}")
        print("[INFO] Press Ctrl+C here to stop all child processes.")

        while True:
            time.sleep(1.0)

            # start_llm_server.py はランチャーなので終了しても正常
            # loop.py もワンショット起動なので終了しても正常
            if combine_proc.poll() is not None:
                print("[WARN] combine_logs.py exited")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C received. Stopping all processes...")
    finally:
        terminate_process_tree(loop_proc, "loop.py")
        terminate_process_tree(combine_proc, "combine_logs.py")
        terminate_process_tree(llm_proc, "start_llm_server.py")

        # 👇 ここ追加
        for port in args.entry_ports:
            kill_process_by_port(port)

        print("[INFO] Shutdown complete.")


if __name__ == "__main__":
    main()