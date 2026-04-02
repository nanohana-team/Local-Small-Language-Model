import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List


DEFAULT_NODES = [
    {
        "name": "A",
        "model": "microsoft/Phi-3.5-mini-instruct",
        "listen_port": 3000,
        "next_ports": [3001],
        "quantization": "8bit",
    },
    {
        "name": "B",
        "model": "LiquidAI/LFM2.5-1.2B-Instruct",
        "listen_port": 3001,
        "next_ports": [3002],
        "quantization": "8bit",
    },
    {
        "name": "C",
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "listen_port": 3002,
        "next_ports": [3000],
        "quantization": "8bit",
    },
]


def build_command(
    python_exe: str,
    script_path: Path,
    model: str,
    listen_port: int,
    next_ports: List[int],
    max_new_tokens: int,
    device: str,
    quantization: str,
) -> List[str]:
    return [
        python_exe,
        str(script_path),
        "--model",
        model,
        "--listen-port",
        str(listen_port),
        "--next-ports",
        *[str(p) for p in next_ports],
        "--max-new-tokens",
        str(max_new_tokens),
        "--device",
        device,
        "--quantization",
        quantization,
    ]


def launch_in_new_console(cmd: List[str], cwd: Path) -> subprocess.Popen:
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_CONSOLE

    return subprocess.Popen(
        cmd,
        cwd=str(cwd),
        creationflags=creationflags,
    )


def resolve_llm_script(chat_dir: Path) -> Path:
    candidates = [
        chat_dir / "llm" / "llm.py",
        chat_dir / "llm.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "llm script not found. looked for: "
        + ", ".join(str(p) for p in candidates)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch multiple LLM bridge servers in separate consoles."
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable to use")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"], help="Inference device for all nodes")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens for all nodes")
    parser.add_argument("--startup-delay", type=float, default=1.0, help="Delay in seconds between launches")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    this_file = Path(__file__).resolve()
    chat_dir = this_file.parent
    llm_script = resolve_llm_script(chat_dir)

    print("=== START LLM SERVERS ===")
    print(f"[INFO] Python     : {args.python}")
    print(f"[INFO] LLM script : {llm_script}")
    print(f"[INFO] Device     : {args.device}")
    print(f"[INFO] Max tokens : {args.max_new_tokens}")

    for node in DEFAULT_NODES:
        cmd = build_command(
            python_exe=args.python,
            script_path=llm_script,
            model=node["model"],
            listen_port=node["listen_port"],
            next_ports=node["next_ports"],
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            quantization=node["quantization"],
        )

        print(
            f"[LAUNCH] {node['name']} | port={node['listen_port']} | "
            f"model={node['model']} | quant={node['quantization']}"
        )
        print(f"[CMD] {' '.join(cmd)}")

        launch_in_new_console(cmd, cwd=chat_dir)
        time.sleep(args.startup_delay)

    print("=== ALL LLM SERVERS LAUNCHED ===")
    print("Nodes:")
    for node in DEFAULT_NODES:
        print(f"  - {node['name']}: 127.0.0.1:{node['listen_port']} -> {node['next_ports']}")


if __name__ == "__main__":
    main()
