from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import threading

from pathlib import Path
from typing import List, Optional

from src.utils.logging import load_dotenv_file, prepare_log_session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local Small Language Model v2 - process orchestrator"
    )

    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--dotenv-path", type=str, default=".env")
    parser.add_argument("--dotenv-override", action="store_true")

    parser.add_argument("--start-local-llm", action="store_true", default=True)
    parser.add_argument("--skip-local-llm", action="store_true")
    parser.add_argument("--local-llm-host", type=str, default="127.0.0.1")
    parser.add_argument("--local-llm-port", type=int, default=8000)
    parser.add_argument("--local-llm-model", type=str, default="")
    parser.add_argument("--local-llm-start-timeout", type=float, default=60.0)

    parser.add_argument("--start-learning", action="store_true", default=True)
    parser.add_argument("--skip-learning", action="store_true")

    parser.add_argument("--start-chat-learning", action="store_true", default=True)
    parser.add_argument("--skip-chat-learning", action="store_true")

    parser.add_argument("--lexicon", type=str, default="libs/dict.lsdx")
    parser.add_argument("--divergence-model", type=str, default="runtime/models/divergence_model.json")
    parser.add_argument("--convergence-model", type=str, default="runtime/models/convergence_model.json")
    parser.add_argument("--verbal-model-path", type=str, default="runtime/models/verbal_model.json")

    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--branch", type=int, default=6)
    parser.add_argument("--final-branch", type=int, default=12)
    parser.add_argument("--update-interval", type=int, default=20)

    parser.add_argument("--input-file", type=str, default="runtime/datasets/teacher.json")
    parser.add_argument("--episode-file", type=str, default="episodes.jsonl")

    parser.add_argument("--gemini-model", type=str, default="gemini-2.5-flash-lite")
    parser.add_argument("--teacher-generator-model", type=str, default="gemini-2.5-flash-lite")
    parser.add_argument("--disable-evaluator", action="store_true")

    parser.add_argument("--auto-teacher", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=100)
    parser.add_argument("--learn-chunk-episodes", type=int, default=100)
    parser.add_argument("--learn-sleep-sec", type=float, default=0.0)

    parser.add_argument("--text", type=str, default="")
    parser.add_argument("--words", nargs="*", default=None)
    parser.add_argument("--max-recursive-steps", type=int, default=6)
    parser.add_argument("--accept-score-threshold", type=float, default=0.85)
    parser.add_argument("--chat-save-path", type=str, default="logs/chat_learning.jsonl")
    parser.add_argument("--chat-sleep-sec", type=float, default=0.0)
    parser.add_argument("--chat-infinity", action="store_true", default=True)

    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _python_cmd() -> List[str]:
    return [sys.executable]


def _make_env(args: argparse.Namespace) -> dict:
    env = dict(os.environ)
    env["LOCAL_LLM_HOST"] = str(args.local_llm_host)
    env["LOCAL_LLM_PORT"] = str(args.local_llm_port)
    if args.local_llm_model.strip():
        env["LOCAL_LLM_MODEL"] = args.local_llm_model.strip()
    return env

def _relay_output(pipe, name: str) -> None:
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            text = line.rstrip("\r\n")
            if text:
                print(f"[{name}] {text}", flush=True)
    except Exception as e:
        print(f"[MAIN][WARN] relay failed for {name}: {e}", flush=True)
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _start_process(cmd: List[str], env: dict, name: str) -> subprocess.Popen:
    print(f"[MAIN] starting {name}: {' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if proc.stdout is not None:
        t = threading.Thread(
            target=_relay_output,
            args=(proc.stdout, name),
            daemon=True,
        )
        t.start()

    return proc


def _wait_http_ready(host: str, port: int, timeout_sec: float) -> bool:
    import json
    import urllib.request
    import urllib.error

    deadline = time.time() + timeout_sec
    urls = [
        f"http://{host}:{port}/health",
        f"http://{host}:{port}/v1/models",
    ]

    while time.time() < deadline:
        for url in urls:
            try:
                with urllib.request.urlopen(url, timeout=2.0) as resp:
                    body = resp.read().decode("utf-8", errors="ignore")
                    if resp.status == 200:
                        if url.endswith("/health"):
                            return True
                        try:
                            json.loads(body)
                            return True
                        except Exception:
                            return True
            except Exception:
                pass
        time.sleep(0.5)
    return False


def build_local_llm_command(args: argparse.Namespace) -> List[str]:
    cmd = _python_cmd() + [
        "-m",
        "src.llm.local_llm_qwen3",
        "--mode",
        "api",
        "--host",
        str(args.local_llm_host),
        "--port",
        str(args.local_llm_port),
    ]
    if args.local_llm_model.strip():
        cmd += ["--model", args.local_llm_model.strip()]
    return cmd


def build_learning_command(args: argparse.Namespace) -> List[str]:
    cmd = _python_cmd() + [
        "-m",
        "src.apps.learning",
        "--lexicon",
        args.lexicon,
        "--divergence-model",
        args.divergence_model,
        "--convergence-model",
        args.convergence_model,
        "--log-dir",
        args.log_dir,
        "--episode-file",
        args.episode_file,
        "--input-file",
        args.input_file,
        "--gemini-model",
        args.gemini_model,
        "--teacher-generator-model",
        args.teacher_generator_model,
        "--depth",
        str(args.depth),
        "--branch",
        str(args.branch),
        "--final-branch",
        str(args.final_branch),
        "--update-interval",
        str(args.update_interval),
        "--max-episodes",
        str(args.max_episodes),
        "--learn-chunk-episodes",
        str(args.learn_chunk_episodes),
        "--learn-sleep-sec",
        str(args.learn_sleep_sec),
        "--dotenv-path",
        args.dotenv_path,
    ]
    if args.dotenv_override:
        cmd.append("--dotenv-override")
    if args.disable_evaluator:
        cmd.append("--disable-evaluator")
    if args.auto_teacher:
        cmd.append("--auto-teacher")
    if args.verbose:
        cmd.append("--verbose")
    return cmd


def build_chat_learning_command(args: argparse.Namespace) -> List[str]:
    cmd = _python_cmd() + [
        "-m",
        "src.apps.chat_learning",
        "--lexicon",
        args.lexicon,
        "--divergence-model",
        args.divergence_model,
        "--convergence-model",
        args.convergence_model,
        "--gemini-model",
        args.gemini_model,
        "--teacher-generator-model",
        args.teacher_generator_model,
        "--depth",
        str(args.depth),
        "--branch",
        str(args.branch),
        "--final-branch",
        str(args.final_branch),
        "--max-recursive-steps",
        str(args.max_recursive_steps),
        "--accept-score-threshold",
        str(args.accept_score_threshold),
        "--input-file",
        args.input_file,
        "--save-path",
        args.chat_save_path,
        "--verbal-model-path",
        args.verbal_model_path,
        "--sleep-sec",
        str(args.chat_sleep_sec),
        "--dotenv-path",
        args.dotenv_path,
    ]
    if args.dotenv_override:
        cmd.append("--dotenv-override")
    if args.disable_evaluator:
        cmd.append("--disable-evaluator")
    if args.auto_teacher:
        cmd.append("--auto-teacher")
    if args.chat_infinity:
        cmd.append("--infinity")
    if args.verbose:
        cmd.append("--verbose")

    if args.words:
        cmd += ["--words", *[str(x) for x in args.words]]
    elif args.text.strip():
        cmd += ["--text", args.text.strip()]

    return cmd


def terminate_process(proc: Optional[subprocess.Popen], name: str) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return

    print(f"[MAIN] terminating {name}", flush=True)

    try:
        if os.name == "nt":
            proc.terminate()
        else:
            proc.send_signal(signal.SIGTERM)
    except Exception:
        pass

    try:
        proc.wait(timeout=10)
        return
    except Exception:
        pass

    try:
        proc.kill()
    except Exception:
        pass
from src.llm.llm_router import load_llm_order

def resolve_local_llm_model_from_config(explicit_model: str) -> str:
    if explicit_model.strip():
        return explicit_model.strip()

    try:
        order = load_llm_order("settings/config.yaml")
    except Exception:
        return ""

    for item in reversed(order):
        s = str(item).strip()
        if s.startswith("local:"):
            return s.split(":", 1)[1].strip()

    return ""

def main() -> None:
    args = parse_args()
    latest_log_path = prepare_log_session(Path(args.log_dir))
    load_dotenv_file(dotenv_path=args.dotenv_path, override=args.dotenv_override)

    env = _make_env(args)

    args.local_llm_model = resolve_local_llm_model_from_config(args.local_llm_model)

    local_llm_proc: Optional[subprocess.Popen] = None
    learning_proc: Optional[subprocess.Popen] = None
    chat_proc: Optional[subprocess.Popen] = None
    try:
        start_local_llm = args.start_local_llm and not args.skip_local_llm
        start_learning = args.start_learning and not args.skip_learning
        start_chat = args.start_chat_learning and not args.skip_chat_learning

        if start_local_llm:
            local_llm_cmd = build_local_llm_command(args)
            local_llm_proc = _start_process(local_llm_cmd, env, "local_llm")

            ready = _wait_http_ready(
                host=args.local_llm_host,
                port=args.local_llm_port,
                timeout_sec=args.local_llm_start_timeout,
            )
            if not ready:
                raise RuntimeError(
                    f"LocalLLM did not become ready within {args.local_llm_start_timeout:.1f}s "
                    f"at {args.local_llm_host}:{args.local_llm_port}"
                )
            print(
                f"[MAIN] LocalLLM ready at http://{args.local_llm_host}:{args.local_llm_port}",
                flush=True,
            )

        if start_learning:
            learning_cmd = build_learning_command(args)
            learning_proc = _start_process(learning_cmd, env, "learning")

        if start_chat:
            chat_cmd = build_chat_learning_command(args)
            chat_proc = _start_process(chat_cmd, env, "chat_learning")

        if not any([local_llm_proc, learning_proc, chat_proc]):
            print("[MAIN] nothing to run", flush=True)
            return

        while True:
            time.sleep(1.0)

            if learning_proc and learning_proc.poll() is not None:
                print(f"[MAIN] learning exited code={learning_proc.returncode}", flush=True)
                learning_proc = None

            if chat_proc and chat_proc.poll() is not None:
                print(f"[MAIN] chat_learning exited code={chat_proc.returncode}", flush=True)
                chat_proc = None

            if local_llm_proc and local_llm_proc.poll() is not None:
                raise RuntimeError(f"LocalLLM exited unexpectedly code={local_llm_proc.returncode}")

            if not any([learning_proc, chat_proc]):
                print("[MAIN] all child apps exited", flush=True)
                break

    except KeyboardInterrupt:
        print("[MAIN] stopped by user", flush=True)
    finally:
        terminate_process(chat_proc, "chat_learning")
        terminate_process(learning_proc, "learning")
        terminate_process(local_llm_proc, "local_llm")

if __name__ == "__main__":
    main()