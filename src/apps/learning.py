from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.apps.learning_central import LearningCentral
from src.core.primitive.divergence import DivergenceModel, DivergencePrimitive
from src.core.primitive.convergence import ConvergenceModel
from src.llm.evaluator_gemini import GeminiEvaluator
from src.llm.input_output_generator_gemini import GeminiTeacherPairGenerator
from src.utils.storage import StorageManager
from src.utils.trainer import Trainer
from src.utils.logging import load_dotenv_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LSLM learning runner")

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
    parser.add_argument("--teacher-generator-model", type=str, default="gemini-2.5-flash-lite")
    parser.add_argument("--disable-evaluator", action="store_true")
    parser.add_argument("--auto-teacher", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=100)
    parser.add_argument("--learn-chunk-episodes", type=int, default=100)
    parser.add_argument("--learn-sleep-sec", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dotenv-path", type=str, default=".env")
    parser.add_argument("--dotenv-override", action="store_true")
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _clean_token_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    return [str(x).strip() for x in values if str(x).strip()]


def _normalize_learning_row(row: Any) -> Optional[Dict[str, Any]]:
    if isinstance(row, list):
        input_tokens = _clean_token_list(row)
        if not input_tokens:
            return None
        return {
            "input_tokens": input_tokens,
            "target_tokens": [],
            "target_text": "",
        }

    if not isinstance(row, dict):
        return None

    input_tokens: List[str] = []
    target_tokens: List[str] = []
    target_text = ""

    if "input_tokens" in row:
        input_tokens = _clean_token_list(row.get("input_tokens"))
    elif "tokens" in row:
        input_tokens = _clean_token_list(row.get("tokens"))
    elif isinstance(row.get("input_text"), str):
        input_tokens = [x for x in str(row["input_text"]).strip().split() if x.strip()]
    elif isinstance(row.get("text"), str):
        input_tokens = [x for x in str(row["text"]).strip().split() if x.strip()]

    if "target_tokens" in row:
        target_tokens = _clean_token_list(row.get("target_tokens"))

    if isinstance(row.get("target_text"), str):
        target_text = str(row.get("target_text", "")).strip()

    if not input_tokens:
        return None

    return {
        "input_tokens": input_tokens,
        "target_tokens": target_tokens,
        "target_text": target_text,
    }


def read_learning_inputs(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        print(f"[WARN] input file not found: {path}", flush=True)
        return []

    episodes: List[Dict[str, Any]] = []
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception as exc:
                    print(f"[WARN] failed to parse jsonl line={line_no}: {exc}", flush=True)
                    continue

                normalized = _normalize_learning_row(row)
                if normalized is not None:
                    episodes.append(normalized)

    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        if isinstance(obj, list):
            for idx, row in enumerate(obj, start=1):
                normalized = _normalize_learning_row(row)
                if normalized is not None:
                    episodes.append(normalized)
                else:
                    print(f"[WARN] skipped invalid json row index={idx}", flush=True)

    else:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                tokens = [x for x in line.strip().split() if x.strip()]
                if tokens:
                    episodes.append({
                        "input_tokens": tokens,
                        "target_tokens": [],
                        "target_text": "",
                    })

    return episodes


def build_app(args: argparse.Namespace) -> LearningCentral:
    lexicon_path = Path(args.lexicon)
    div_model_path = Path(args.divergence_model)
    conv_model_path = Path(args.convergence_model)
    log_dir = Path(args.log_dir)
    episode_path = log_dir / args.episode_file

    ensure_parent(div_model_path)
    ensure_parent(conv_model_path)
    ensure_parent(episode_path)

    print(f"[LEXICON] loading from {lexicon_path}", flush=True)
    lexicon = DivergencePrimitive.load_lexicon(lexicon_path)
    print(f"[LEXICON] loaded entries={len(lexicon)}", flush=True)

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

    teacher_generator = GeminiTeacherPairGenerator(
        model_name=args.teacher_generator_model,
        enabled=args.auto_teacher,
        verbose=args.verbose,
    )

    storage = StorageManager(
        log_dir=log_dir,
        episode_file=args.episode_file,
    )

    trainer = Trainer(
        divergence_model_path=div_model_path,
        convergence_model_path=conv_model_path,
        dict_path=None,
        enable_unknown_token_expansion=False,
        gemini_model_name=args.gemini_model,
    )

    return LearningCentral(
        divergence_model=divergence_model,
        convergence_model=convergence_model,
        evaluator=evaluator,
        storage=storage,
        trainer=trainer,
        verbose=args.verbose,
        input_generator=teacher_generator,
    )


def run_infinite_learning_mode(app: LearningCentral, args: argparse.Namespace) -> int:
    dataset: Optional[List[Dict[str, Any]]] = None

    if not args.auto_teacher:
        dataset = read_learning_inputs(Path(args.input_file))
        if not dataset:
            print("[WARN] no learning inputs found. nothing to do.", flush=True)
            return 0

        supervised_count = sum(
            1 for row in dataset
            if row.get("target_tokens") or str(row.get("target_text", "")).strip()
        )
        print(
            f"[LEARN] dataset loaded rows={len(dataset)} supervised={supervised_count} unsupervised={len(dataset) - supervised_count}",
            flush=True,
        )

    chunk_episodes = max(1, int(args.learn_chunk_episodes))
    chunk_index = 0

    print("[LEARN] infinite learning loop started", flush=True)
    print("[LEARN] stop with Ctrl+C", flush=True)

    try:
        while True:
            chunk_index += 1
            print(
                f"[LEARN] chunk={chunk_index} "
                f"episodes={chunk_episodes} "
                f"auto_teacher={'on' if args.auto_teacher else 'off'}",
                flush=True,
            )

            app.run_learning_loop(
                dataset=None if args.auto_teacher else dataset,
                depth=args.depth,
                update_interval=args.update_interval,
                max_episodes=chunk_episodes,
                auto_teacher=args.auto_teacher,
            )

            if args.learn_sleep_sec > 0:
                time.sleep(args.learn_sleep_sec)

    except KeyboardInterrupt:
        print("[LEARN] stopped by user", flush=True)
        return 0


def main() -> None:
    args = parse_args()

    load_dotenv_file(dotenv_path=args.dotenv_path, override=args.dotenv_override)

    has_gemini = bool(os.getenv("GEMINI_API_KEY", "").strip())
    has_openai = bool(os.getenv("OPENAI_API_KEY", "").strip())
    local_host = os.getenv("LOCAL_LLM_HOST", "127.0.0.1")
    local_port = os.getenv("LOCAL_LLM_PORT", "8000")

    print(f"[ENV] GEMINI_API_KEY={'set' if has_gemini else 'missing'}", flush=True)
    print(f"[ENV] OPENAI_API_KEY={'set' if has_openai else 'missing'}", flush=True)
    print(f"[ENV] LOCAL_LLM={local_host}:{local_port}", flush=True)

    app = build_app(args)
    raise SystemExit(run_infinite_learning_mode(app, args))


if __name__ == "__main__":
    main()