import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

DEFAULT_NODES = [
    {"name": "A", "model": "microsoft/Phi-3.5-mini-instruct", "listen_port": 3000, "next_ports": [3001], "quantization": "8bit"},
    {"name": "B", "model": "LiquidAI/LFM2.5-1.2B-Instruct", "listen_port": 3001, "next_ports": [3002], "quantization": "8bit"},
    {"name": "C", "model": "Qwen/Qwen2.5-1.5B-Instruct", "listen_port": 3002, "next_ports": [3000], "quantization": "8bit"},
]

def build_command(python_exe: str, script_path: Path, model: str, listen_port: int, next_ports: List[int], max_new_tokens: int, device: str, quantization: str) -> List[str]:
    return [python_exe, str(script_path), "--model", model, "--listen-port", str(listen_port), "--next-ports", *[str(p) for p in next_ports], "--max-new-tokens", str(max_new_tokens), "--device", device, "--quantization", quantization]

def resolve_llm_script(chat_dir: Path) -> Path:
    for candidate in [chat_dir / "llm" / "llm.py", chat_dir / "llm.py"]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("llm script not found")

def launcher_log_dir(chat_dir: Path) -> Path:
    path = chat_dir / "logs" / "launcher" / time.strftime("%Y%m%d_%H%M%S")
    path.mkdir(parents=True, exist_ok=True)
    return path

def launch_in_new_console(cmd: List[str], cwd: Path, log_file: Path) -> subprocess.Popen:
    if os.name == "nt":
        quoted = subprocess.list2cmdline(cmd)
        full_cmd = f'{quoted} 1>> "{log_file}" 2>&1'
        return subprocess.Popen(["cmd.exe", "/k", full_cmd], cwd=str(cwd), creationflags=subprocess.CREATE_NEW_CONSOLE)
    with log_file.open("a", encoding="utf-8") as fh:
        return subprocess.Popen(cmd, cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch multiple LLM bridge servers in separate consoles.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--startup-delay", type=float, default=1.0)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    chat_dir = Path(__file__).resolve().parent
    llm_script = resolve_llm_script(chat_dir)
    logs_dir = launcher_log_dir(chat_dir)
    manifest = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "python": args.python, "device": args.device, "max_new_tokens": args.max_new_tokens, "llm_script": str(llm_script), "nodes": []}
    print(f"[INFO] Launch logs: {logs_dir}")
    for node in DEFAULT_NODES:
        cmd = build_command(args.python, llm_script, node["model"], node["listen_port"], node["next_ports"], args.max_new_tokens, args.device, node["quantization"])
        log_file = logs_dir / f"node_{node['name']}_port_{node['listen_port']}.log"
        print(f"[LAUNCH] {node['name']} | port={node['listen_port']} | model={node['model']} | quant={node['quantization']}")
        proc = launch_in_new_console(cmd, cwd=chat_dir, log_file=log_file)
        manifest["nodes"].append({**node, "pid": proc.pid, "log_file": str(log_file)})
        time.sleep(args.startup_delay)
    (logs_dir / "launch_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
