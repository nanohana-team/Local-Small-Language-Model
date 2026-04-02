import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, row: Dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_command(
    cmd: List[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> None:
    print("")
    print("[RUN]", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


@dataclass
class LoopConfig:
    python_exe: str
    project_root: str
    iterations: int
    base_model: str
    prompts_file: str
    teacher_config: str
    default_system_prompt: str
    output_root: str
    repeats_per_teacher: int
    candidate_limit: int
    sleep_sec: float
    train_epochs: float
    train_batch_size: int
    eval_batch_size: int
    grad_accum: int
    learning_rate: float
    max_length: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    bf16: bool
    fp16: bool
    gradient_checkpointing: bool
    merge_after_training: bool
    dry_run_generation: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iterative SFT loop for Gemma 3 270M student model."
    )
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--iterations", type=int, default=3)

    parser.add_argument("--base-model", type=str, default="google/gemma-3-270m-it")
    parser.add_argument("--prompts-file", type=str, default="data/prompts/input_prompts.jsonl")
    parser.add_argument("--teacher-config", type=str, default="config/teachers.json")
    parser.add_argument(
        "--default-system-prompt",
        type=str,
        default="あなたは日本語で自然に会話するAIです。簡潔で自然な返答をしてください。",
    )

    parser.add_argument("--output-root", type=str, default="runs/sft_loop")
    parser.add_argument("--repeats-per-teacher", type=int, default=2)
    parser.add_argument("--candidate-limit", type=int, default=0)
    parser.add_argument("--sleep-sec", type=float, default=0.2)
    parser.add_argument("--dry-run-generation", action="store_true")

    parser.add_argument("--train-epochs", type=float, default=2.0)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--merge-after-training", action="store_true")

    return parser.parse_args()


def build_loop_config(args: argparse.Namespace) -> LoopConfig:
    return LoopConfig(
        python_exe=args.python_exe,
        project_root=args.project_root,
        iterations=args.iterations,
        base_model=args.base_model,
        prompts_file=args.prompts_file,
        teacher_config=args.teacher_config,
        default_system_prompt=args.default_system_prompt,
        output_root=args.output_root,
        repeats_per_teacher=args.repeats_per_teacher,
        candidate_limit=args.candidate_limit,
        sleep_sec=args.sleep_sec,
        train_epochs=args.train_epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        merge_after_training=args.merge_after_training,
        dry_run_generation=args.dry_run_generation,
    )


def resolve_model_for_next_iteration(
    previous_merged_dir: Path,
    previous_lora_dir: Path,
    fallback_model_name: str,
    merge_after_training: bool,
) -> str:
    if merge_after_training and previous_merged_dir.exists():
        return str(previous_merged_dir)
    if previous_lora_dir.exists():
        return str(previous_lora_dir)
    return fallback_model_name


def main() -> int:
    args = parse_args()
    cfg = build_loop_config(args)

    project_root = Path(cfg.project_root).resolve()
    output_root = (project_root / cfg.output_root).resolve()
    ensure_dir(output_root)

    run_id = f"{now_stamp()}_{uuid.uuid4().hex[:8]}"
    run_root = output_root / run_id
    ensure_dir(run_root)

    manifest_path = run_root / "manifest.json"
    summary_log_path = run_root / "summary.jsonl"

    write_json(
        manifest_path,
        {
            "run_id": run_id,
            "started_at": now_stamp(),
            "config": asdict(cfg),
        },
    )

    current_student_model = cfg.base_model

    print("[INFO] run_id =", run_id)
    print("[INFO] run_root =", run_root)
    print("[INFO] base_model =", current_student_model)

    for iteration in range(1, cfg.iterations + 1):
        iter_name = f"iter_{iteration:03d}"
        iter_root = run_root / iter_name
        ensure_dir(iter_root)

        candidates_file = iter_root / "candidates.jsonl"
        scored_file = iter_root / "scored_candidates.jsonl"
        sft_train_file = iter_root / "sft_train.jsonl"
        sft_eval_file = iter_root / "sft_eval.jsonl"

        lora_out_dir = iter_root / "student_lora"
        merged_out_dir = iter_root / "student_merged"

        print("")
        print("=" * 80)
        print(f"[INFO] ITERATION {iteration}/{cfg.iterations}")
        print(f"[INFO] current_student_model={current_student_model}")
        print("=" * 80)

        iter_started = time.time()

        # 1) generate candidates from teacher models
        gen_cmd = [
            cfg.python_exe,
            str(project_root / "src" / "data" / "generate_candidates.py"),
            "--input-file",
            str(project_root / cfg.prompts_file),
            "--teacher-config",
            str(project_root / cfg.teacher_config),
            "--output-file",
            str(candidates_file),
            "--continue-on-error",
            "--sleep-sec",
            str(cfg.sleep_sec),
            "--repeats-per-teacher",
            str(cfg.repeats_per_teacher),
            "--default-system-prompt",
            cfg.default_system_prompt,
        ]
        if cfg.candidate_limit > 0:
            gen_cmd.extend(["--limit", str(cfg.candidate_limit)])
        if cfg.dry_run_generation:
            gen_cmd.append("--dry-run")

        run_command(gen_cmd, cwd=project_root)

        # 2) score candidates
        score_cmd = [
            cfg.python_exe,
            str(project_root / "src" / "eval" / "score_candidates.py"),
            "--input-file",
            str(candidates_file),
            "--output-file",
            str(scored_file),
        ]
        run_command(score_cmd, cwd=project_root)

        # 3) build SFT dataset
        build_cmd = [
            cfg.python_exe,
            str(project_root / "src" / "data" / "build_sft_dataset.py"),
            "--input-file",
            str(scored_file),
            "--train-output-file",
            str(sft_train_file),
            "--eval-output-file",
            str(sft_eval_file),
        ]
        run_command(build_cmd, cwd=project_root)

        # 4) train student
        train_cmd = [
            cfg.python_exe,
            str(project_root / "src" / "train" / "train_gemma3_270m_sft.py"),
            "--model-name",
            current_student_model,
            "--train-file",
            str(sft_train_file),
            "--eval-file",
            str(sft_eval_file),
            "--output-dir",
            str(lora_out_dir),
            "--merged-output-dir",
            str(merged_out_dir),
            "--max-length",
            str(cfg.max_length),
            "--per-device-train-batch-size",
            str(cfg.train_batch_size),
            "--per-device-eval-batch-size",
            str(cfg.eval_batch_size),
            "--gradient-accumulation-steps",
            str(cfg.grad_accum),
            "--num-train-epochs",
            str(cfg.train_epochs),
            "--learning-rate",
            str(cfg.learning_rate),
            "--lora-r",
            str(cfg.lora_r),
            "--lora-alpha",
            str(cfg.lora_alpha),
            "--lora-dropout",
            str(cfg.lora_dropout),
        ]

        if cfg.bf16:
            train_cmd.append("--bf16")
        if cfg.fp16:
            train_cmd.append("--fp16")
        if cfg.gradient_checkpointing:
            train_cmd.append("--gradient-checkpointing")
        if cfg.merge_after_training:
            train_cmd.append("--merge-after-training")

        run_command(train_cmd, cwd=project_root)

        iter_elapsed = round(time.time() - iter_started, 2)

        next_student_model = resolve_model_for_next_iteration(
            previous_merged_dir=merged_out_dir,
            previous_lora_dir=lora_out_dir,
            fallback_model_name=current_student_model,
            merge_after_training=cfg.merge_after_training,
        )

        summary_row = {
            "run_id": run_id,
            "iteration": iteration,
            "iter_root": str(iter_root),
            "input_model": current_student_model,
            "output_lora_dir": str(lora_out_dir),
            "output_merged_dir": str(merged_out_dir),
            "next_model": next_student_model,
            "elapsed_sec": iter_elapsed,
            "artifacts": {
                "candidates_file": str(candidates_file),
                "scored_file": str(scored_file),
                "sft_train_file": str(sft_train_file),
                "sft_eval_file": str(sft_eval_file),
            },
        }
        append_jsonl(summary_log_path, summary_row)

        current_student_model = next_student_model

        print(f"[INFO] iteration {iteration} finished in {iter_elapsed}s")
        print(f"[INFO] next_student_model = {current_student_model}")

    final_info = {
        "run_id": run_id,
        "finished_at": now_stamp(),
        "final_model": current_student_model,
        "summary_log": str(summary_log_path),
    }
    write_json(run_root / "final.json", final_info)

    print("")
    print("[INFO] ALL DONE")
    print("[INFO] final_model =", current_student_model)
    print("[INFO] summary_log =", summary_log_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())