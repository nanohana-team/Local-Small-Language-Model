# main.py
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

from src.apps.learning_central import LearningCentral
from src.core.primitive.divergence import DivergenceModel, DivergencePrimitive
from src.core.primitive.convergence import ConvergenceModel
from src.llm.evaluator_gemini import GeminiEvaluator
from src.llm.input_generator_gemini import GeminiInputGenerator
from src.utils.storage import StorageManager
from src.utils.trainer import Trainer
from src.apps.chat_control import ChatController


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

        if value and len(value) >= 2:
            if (value[0] == value[-1]) and value[0] in {'"', "'"}:
                value = value[1:-1]

        if override or key not in os.environ:
            os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local Small Language Model v2 - learning/chat entrypoint"
    )

    parser.add_argument("--mode", choices=["learn", "learning", "chat"], required=True)
    parser.add_argument("--lexicon", type=str, default="libs/dict.lsdx")
    parser.add_argument("--divergence-model", type=str, default="runtime/models/divergence_model.json")
    parser.add_argument("--convergence-model", type=str, default="runtime/models/convergence_model.json")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--branch", type=int, default=6)
    parser.add_argument("--final-branch", type=int, default=12)
    parser.add_argument("--update-interval", type=int, default=20)
    parser.add_argument("--log-dir", type=str, default="runtime/logs")
    parser.add_argument("--episode-file", type=str, default="episodes.jsonl")
    parser.add_argument("--input-file", type=str, default="runtime/datasets/inputs.jsonl")
    parser.add_argument("--gemini-model", type=str, default="gemini-2.5-flash-lite")
    parser.add_argument("--input-generator-model", type=str, default="gemini-2.5-flash-lite")
    parser.add_argument("--disable-evaluator", action="store_true")
    parser.add_argument("--auto-input", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=100)
    parser.add_argument("--learn-chunk-episodes", type=int, default=100)
    parser.add_argument("--learn-sleep-sec", type=float, default=0.0)
    parser.add_argument("--text", type=str, default="")
    parser.add_argument("--words", nargs="*", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dotenv-path", type=str, default=".env")
    parser.add_argument("--dotenv-override", action="store_true")
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_learning_inputs(path: Path) -> List[List[str]]:
    if not path.exists():
        print(f"[WARN] input file not found: {path}")
        return []

    episodes: List[List[str]] = []
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    if "tokens" in row and isinstance(row["tokens"], list):
                        episodes.append([str(x) for x in row["tokens"] if str(x).strip()])
                    elif "text" in row and isinstance(row["text"], str):
                        episodes.append(row["text"].strip().split())
                elif isinstance(row, list):
                    episodes.append([str(x) for x in row if str(x).strip()])

    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        if isinstance(obj, list):
            for row in obj:
                if isinstance(row, dict):
                    if "tokens" in row and isinstance(row["tokens"], list):
                        episodes.append([str(x) for x in row["tokens"] if str(x).strip()])
                    elif "text" in row and isinstance(row["text"], str):
                        episodes.append(row["text"].strip().split())
                elif isinstance(row, list):
                    episodes.append([str(x) for x in row if str(x).strip()])
    else:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if tokens:
                    episodes.append(tokens)

    return episodes


def print_gemini_env_status(auto_input: bool, evaluator_enabled: bool) -> None:
    has_key = bool(os.getenv("GEMINI_API_KEY", "").strip())
    if auto_input or evaluator_enabled:
        print(f"[ENV] GEMINI_API_KEY={'set' if has_key else 'missing'}")


def build_chat_controller(args: argparse.Namespace) -> ChatController:
    lexicon_path = Path(args.lexicon)
    div_model_path = Path(args.divergence_model)
    conv_model_path = Path(args.convergence_model)

    ensure_parent(div_model_path)
    ensure_parent(conv_model_path)

    print(f"[LEXICON] loading from {lexicon_path}")
    lexicon = DivergencePrimitive.load_lexicon(lexicon_path)
    print(f"[LEXICON] loaded entries={len(lexicon)}")

    divergence_model = DivergenceModel(
        lexicon=lexicon,
        model_path=div_model_path,
        default_branch=args.branch,
        final_branch=args.final_branch,
    )

    convergence_model = ConvergenceModel(
        divergence_model=divergence_model,
        model_path=conv_model_path,
    )

    evaluator = GeminiEvaluator(
        model_name=args.gemini_model,
        enabled=not args.disable_evaluator,
        verbose=args.verbose,
    )

    controller = ChatController(
        divergence_model=divergence_model,
        convergence_model=convergence_model,
        evaluator=evaluator,
        storage=None,
        trainer=None,
        verbose=args.verbose,
    )
    return controller


def build_app(args: argparse.Namespace) -> LearningCentral:
    lexicon_path = Path(args.lexicon)
    div_model_path = Path(args.divergence_model)
    conv_model_path = Path(args.convergence_model)
    log_dir = Path(args.log_dir)
    episode_path = log_dir / args.episode_file

    ensure_parent(div_model_path)
    ensure_parent(conv_model_path)
    ensure_parent(episode_path)

    print(f"[LEXICON] loading from {lexicon_path}")
    lexicon = DivergencePrimitive.load_lexicon(lexicon_path)
    print(f"[LEXICON] loaded entries={len(lexicon)}")

    divergence_model = DivergenceModel(
        lexicon=lexicon,
        model_path=div_model_path,
        default_branch=args.branch,
        final_branch=args.final_branch,
    )

    convergence_model = ConvergenceModel(
        divergence_model=divergence_model,
        model_path=conv_model_path,
    )

    evaluator = GeminiEvaluator(
        model_name=args.gemini_model,
        enabled=not args.disable_evaluator,
        verbose=args.verbose,
    )

    input_generator = GeminiInputGenerator(
        model_name=args.input_generator_model,
        enabled=args.auto_input,
    )

    storage = StorageManager(
        log_dir=log_dir,
        episode_file=args.episode_file,
    )

    trainer = Trainer(
        divergence_model_path=div_model_path,
        convergence_model_path=conv_model_path,
        dict_path=None,  # .lsdx は UnknownTokenExpander に渡さない
        enable_unknown_token_expansion=False,
        gemini_model_name=args.gemini_model,
    )

    app = LearningCentral(
        divergence_model=divergence_model,
        convergence_model=convergence_model,
        evaluator=evaluator,
        storage=storage,
        trainer=trainer,
        verbose=args.verbose,
        input_generator=input_generator,
    )

    return app


def run_chat_mode(controller: ChatController, args: argparse.Namespace) -> int:
    if args.words:
        input_tokens = [str(x) for x in args.words if str(x).strip()]
    elif args.text.strip():
        input_tokens = args.text.strip().split()
    else:
        print("[ERROR] chat mode requires --text or --words")
        return 1

    result = controller.run_once(
        input_tokens=input_tokens,
        depth=args.depth,
    )

    print("\n[FINAL RESPONSE]")
    print(result.get("response_text", ""))

    if args.verbose:
        print("\n[DETAIL]")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    return 0


def run_learning_mode(app: LearningCentral, args: argparse.Namespace) -> int:
    if args.auto_input:
        app.run_learning_loop(
            dataset=None,
            depth=args.depth,
            update_interval=args.update_interval,
            max_episodes=args.max_episodes,
            auto_input=True,
        )
        return 0

    learning_inputs = read_learning_inputs(Path(args.input_file))
    if not learning_inputs:
        print("[WARN] no learning inputs found. nothing to do.")
        return 0

    app.run_learning_loop(
        dataset=learning_inputs,
        depth=args.depth,
        update_interval=args.update_interval,
        max_episodes=args.max_episodes,
        auto_input=False,
    )
    return 0


def run_infinite_learning_mode(app: LearningCentral, args: argparse.Namespace) -> int:
    dataset: Optional[List[List[str]]] = None

    if not args.auto_input:
        dataset = read_learning_inputs(Path(args.input_file))
        if not dataset:
            print("[WARN] no learning inputs found. nothing to do.")
            return 0

    chunk_episodes = max(1, int(args.learn_chunk_episodes))
    chunk_index = 0

    print("[LEARN] infinite learning loop started")
    print("[LEARN] stop with Ctrl+C")

    try:
        while True:
            chunk_index += 1
            print(
                f"[LEARN] chunk={chunk_index} "
                f"episodes={chunk_episodes} "
                f"auto_input={'on' if args.auto_input else 'off'}"
            )

            app.run_learning_loop(
                dataset=None if args.auto_input else dataset,
                depth=args.depth,
                update_interval=args.update_interval,
                max_episodes=chunk_episodes,
                auto_input=args.auto_input,
            )

            if args.learn_sleep_sec > 0:
                time.sleep(args.learn_sleep_sec)

    except KeyboardInterrupt:
        print("\n[LEARN] stopped by user")
        return 0


def main() -> None:
    args = parse_args()

    load_dotenv_file(
        dotenv_path=args.dotenv_path,
        override=args.dotenv_override,
    )

    print_gemini_env_status(
        auto_input=args.auto_input,
        evaluator_enabled=not args.disable_evaluator,
    )

    if args.mode == "chat":
        controller = build_chat_controller(args)
        code = run_chat_mode(controller, args)
        sys.exit(code)

    app = build_app(args)

    if args.mode == "learning":
        code = run_learning_mode(app, args)
        sys.exit(code)

    if args.mode == "learn":
        code = run_infinite_learning_mode(app, args)
        sys.exit(code)

    print(f"[ERROR] unsupported mode: {args.mode}")
    sys.exit(1)


if __name__ == "__main__":
    main()