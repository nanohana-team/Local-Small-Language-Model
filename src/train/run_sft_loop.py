"""
会話ベース反復SFTループ (architecture.md 準拠)

各イテレーションで以下を実行する:
  1. 複数モデル会話  : teacher 3体 + learning 1体がランダム順で会話
  2. Gemini評価     : 会話セッション全体を Gemini API で採点
  3. SFTデータ生成  : 高スコアな learning 発話を訓練データに変換
  4. LoRA学習       : learning モデルを SFT で再学習
  5. 次イテレーション: 更新されたモデルで再び会話

起動例:
  python src/train/run_sft_loop.py \
      --iterations 3 \
      --bf16 \
      --merge-after-training
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import argparse


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, row: Dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_command(cmd: List[str], cwd: Optional[Path] = None) -> None:
    """コマンドを実行し、失敗したら RuntimeError を送出する。"""
    print("\n" + "─" * 60)
    print("[RUN] " + " ".join(str(x) for x in cmd))
    print("─" * 60)
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}):\n  " + " ".join(str(x) for x in cmd)
        )


def write_learning_config(path: Path, base_cfg: Dict[str, Any], model_path: str) -> None:
    """base_cfg をベースに model フィールドだけ差し替えて書き出す。"""
    cfg = dict(base_cfg)
    cfg["model"] = model_path
    write_json(path, cfg)


def resolve_next_model(
    merged_dir: Path,
    lora_dir: Path,
    fallback: str,
    merge: bool,
) -> str:
    """次イテレーションで使うモデルパスを決定する。"""
    if merge and merged_dir.exists() and any(merged_dir.iterdir()):
        return str(merged_dir)
    if lora_dir.exists() and any(lora_dir.iterdir()):
        return str(lora_dir)
    return fallback


# ---------------------------------------------------------------------------
# 引数
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "会話ベース反復SFTループ: "
            "多モデル会話 → Gemini評価 → SFTデータ生成 → LoRA学習 を繰り返す"
        )
    )

    # 実行環境
    p.add_argument("--python-exe", default=sys.executable, help="使用する Python 実行ファイル")
    p.add_argument("--project-root", default=".", help="プロジェクトルートディレクトリ")
    p.add_argument("--iterations", type=int, default=3, help="ループ回数")

    # モデル / 設定ファイル
    p.add_argument("--base-model", default="google/gemma-3-270m-it",
                   help="learning モデルの初期ベースモデル (HuggingFace ID またはローカルパス)")
    p.add_argument("--prompts-file", default="data/prompts/input_prompts.jsonl",
                   help="会話の起点となるプロンプト JSONL")
    p.add_argument("--teachers-config", default="config/teachers.json",
                   help="teacher モデル設定 JSON")
    p.add_argument("--learning-config", default="config/learning.json",
                   help="learning モデル設定 JSON (model フィールドはループ内で上書き)")
    p.add_argument("--loop-config", default="config/conversation_loop.json",
                   help="会話ループ設定 JSON (ラウンド数・参加者リスト等)")
    p.add_argument("--schema-file", default="config/gemini_eval_schema.json",
                   help="Gemini 評価レスポンスのスキーマ JSON")
    p.add_argument("--env-file", default=".env",
                   help="GEMINI_API_KEY 等を含む .env ファイルのパス")
    p.add_argument("--learning-name", default="learning_gemma",
                   help="learning モデルの参加者名 (learning.json の name と一致させること)")

    # 出力
    p.add_argument("--output-root", default="runs/sft_loop",
                   help="ループ実行結果の保存先ルートディレクトリ")

    # 会話生成オプション
    p.add_argument("--prompt-limit", type=int, default=0,
                   help="処理するプロンプト数の上限 (0=全件)")

    # データセット構築オプション
    p.add_argument("--min-overall", type=float, default=0.70,
                   help="SFT採用の最低 overall スコア (0.0〜1.0)")
    p.add_argument("--eval-ratio", type=float, default=0.05,
                   help="eval 分割割合")

    # 学習ハイパーパラメータ
    p.add_argument("--train-epochs", type=float, default=2.0)
    p.add_argument("--train-batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--max-length", type=int, default=512)

    # LoRA
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    # 精度 / その他
    p.add_argument("--bf16", action="store_true", help="bfloat16 で学習")
    p.add_argument("--fp16", action="store_true", help="float16 で学習")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--merge-after-training", action="store_true",
                   help="学習後に LoRA をベースモデルにマージする")

    return p.parse_args()


# ---------------------------------------------------------------------------
# ステップ実装
# ---------------------------------------------------------------------------

def step_conversation(
    args: argparse.Namespace,
    project_root: Path,
    learning_cfg_file: Path,
    sessions_file: Path,
) -> None:
    """Step 1: teacher + learning によるランダム順会話セッション生成"""
    cmd = [
        args.python_exe,
        str(project_root / "src" / "loop" / "run_conversation_loop.py"),
        "--prompts-file",    str(project_root / args.prompts_file),
        "--teachers-config", str(project_root / args.teachers_config),
        "--learning-config", str(learning_cfg_file),
        "--loop-config",     str(project_root / args.loop_config),
        "--output-file",     str(sessions_file),
    ]
    if args.prompt_limit > 0:
        cmd.extend(["--limit", str(args.prompt_limit)])
    run_command(cmd, cwd=project_root)


def step_gemini_eval(
    args: argparse.Namespace,
    project_root: Path,
    sessions_file: Path,
    evaluated_file: Path,
) -> None:
    """Step 2: Gemini API による会話セッション評価"""
    cmd = [
        args.python_exe,
        str(project_root / "src" / "eval" / "evaluate_conversations_with_gemini.py"),
        "--input-file",    str(sessions_file),
        "--output-file",   str(evaluated_file),
        "--learning-name", args.learning_name,
        "--schema-file",   str(project_root / args.schema_file),
        "--env-file",      str(project_root / args.env_file),
    ]
    run_command(cmd, cwd=project_root)


def step_build_sft(
    args: argparse.Namespace,
    project_root: Path,
    evaluated_file: Path,
    sft_train_file: Path,
    sft_eval_file: Path,
) -> None:
    """Step 3: 高スコアな learning 発話から SFT データセットを構築"""
    cmd = [
        args.python_exe,
        str(project_root / "src" / "data" / "build_sft_dataset_from_sessions.py"),
        "--input-file",        str(evaluated_file),
        "--train-output-file", str(sft_train_file),
        "--eval-output-file",  str(sft_eval_file),
        "--min-overall",       str(args.min_overall),
        "--eval-ratio",        str(args.eval_ratio),
    ]
    run_command(cmd, cwd=project_root)


def step_train(
    args: argparse.Namespace,
    project_root: Path,
    current_model: str,
    sft_train_file: Path,
    sft_eval_file: Path,
    lora_out_dir: Path,
    merged_out_dir: Path,
) -> None:
    """Step 4: LoRA SFT 学習"""
    cmd = [
        args.python_exe,
        str(project_root / "src" / "train" / "train_gemma3_270m_sft.py"),
        "--model-name",                  current_model,
        "--train-file",                  str(sft_train_file),
        "--eval-file",                   str(sft_eval_file),
        "--output-dir",                  str(lora_out_dir),
        "--merged-output-dir",           str(merged_out_dir),
        "--max-length",                  str(args.max_length),
        "--per-device-train-batch-size", str(args.train_batch_size),
        "--per-device-eval-batch-size",  str(args.eval_batch_size),
        "--gradient-accumulation-steps", str(args.grad_accum),
        "--num-train-epochs",            str(args.train_epochs),
        "--learning-rate",               str(args.learning_rate),
        "--lora-r",                      str(args.lora_r),
        "--lora-alpha",                  str(args.lora_alpha),
        "--lora-dropout",                str(args.lora_dropout),
    ]
    if args.bf16:
        cmd.append("--bf16")
    if args.fp16:
        cmd.append("--fp16")
    if args.gradient_checkpointing:
        cmd.append("--gradient-checkpointing")
    if args.merge_after_training:
        cmd.append("--merge-after-training")
    run_command(cmd, cwd=project_root)


# ---------------------------------------------------------------------------
# メインループ
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    project_root = Path(args.project_root).resolve()
    output_root = (project_root / args.output_root).resolve()
    ensure_dir(output_root)

    run_id = f"{now_stamp()}_{uuid.uuid4().hex[:8]}"
    run_root = output_root / run_id
    ensure_dir(run_root)

    # ベース learning config を読み込み（model フィールドはイテレーションごとに上書き）
    with (project_root / args.learning_config).open("r", encoding="utf-8") as f:
        base_learning_cfg: Dict[str, Any] = json.load(f)

    # マニフェスト書き出し
    write_json(run_root / "manifest.json", {
        "run_id": run_id,
        "started_at": now_stamp(),
        "pipeline": [
            "1. multi-model conversation (random order)",
            "2. Gemini evaluation",
            "3. SFT dataset from sessions",
            "4. LoRA fine-tuning",
        ],
        "args": vars(args),
    })

    current_model = args.base_model
    summary_log = run_root / "summary.jsonl"

    print("=" * 70)
    print(f"[INFO] run_id     = {run_id}")
    print(f"[INFO] run_root   = {run_root}")
    print(f"[INFO] base_model = {current_model}")
    print(f"[INFO] iterations = {args.iterations}")
    print("=" * 70)

    for iteration in range(1, args.iterations + 1):
        iter_name = f"iter_{iteration:03d}"
        iter_root = run_root / iter_name
        ensure_dir(iter_root)

        print(f"\n{'#' * 70}")
        print(f"# ITERATION {iteration}/{args.iterations}")
        print(f"# current model: {current_model}")
        print(f"{'#' * 70}")

        iter_started = time.time()

        # イテレーションごとのファイルパス
        learning_cfg_file = iter_root / "learning.json"   # model を上書きした設定
        sessions_file     = iter_root / "sessions.jsonl"   # 会話セッション
        evaluated_file    = iter_root / "evaluated.jsonl"  # Gemini評価結果
        sft_train_file    = iter_root / "sft_train.jsonl"  # 学習データ
        sft_eval_file     = iter_root / "sft_eval.jsonl"   # 評価データ
        lora_out_dir      = iter_root / "student_lora"     # LoRAアダプタ
        merged_out_dir    = iter_root / "student_merged"   # マージ済みモデル

        try:
            # Step 1: イテレーション用 learning config 書き出し
            write_learning_config(learning_cfg_file, base_learning_cfg, current_model)
            print(f"\n[Step 1/4] Multi-model conversation  (random_order=True)")
            step_conversation(args, project_root, learning_cfg_file, sessions_file)

            # Step 2: Gemini 評価
            print(f"\n[Step 2/4] Gemini evaluation")
            step_gemini_eval(args, project_root, sessions_file, evaluated_file)

            # Step 3: SFT データセット構築
            print(f"\n[Step 3/4] Build SFT dataset  (min_overall={args.min_overall})")
            step_build_sft(args, project_root, evaluated_file, sft_train_file, sft_eval_file)

            # Step 4: LoRA 学習
            print(f"\n[Step 4/4] LoRA fine-tuning")
            step_train(
                args, project_root, current_model,
                sft_train_file, sft_eval_file,
                lora_out_dir, merged_out_dir,
            )

        except RuntimeError as exc:
            print(f"\n[ERROR] iteration {iteration} failed: {exc}")
            append_jsonl(summary_log, {
                "iteration": iteration,
                "status": "failed",
                "error": str(exc),
                "input_model": current_model,
            })
            return 1

        # 次イテレーションのモデルを決定
        next_model = resolve_next_model(
            merged_out_dir, lora_out_dir, current_model, args.merge_after_training
        )
        elapsed = round(time.time() - iter_started, 2)

        append_jsonl(summary_log, {
            "iteration": iteration,
            "status": "ok",
            "input_model": current_model,
            "next_model": next_model,
            "elapsed_sec": elapsed,
            "files": {
                "sessions":    str(sessions_file),
                "evaluated":   str(evaluated_file),
                "sft_train":   str(sft_train_file),
                "sft_eval":    str(sft_eval_file),
                "lora_dir":    str(lora_out_dir),
                "merged_dir":  str(merged_out_dir),
            },
        })

        current_model = next_model
        print(f"\n[INFO] iteration {iteration} complete  elapsed={elapsed}s")
        print(f"[INFO] next model: {current_model}")

    # 完了
    write_json(run_root / "final.json", {
        "run_id": run_id,
        "finished_at": now_stamp(),
        "final_model": current_model,
        "summary_log": str(summary_log),
    })

    print(f"\n{'=' * 70}")
    print(f"[INFO] ALL DONE")
    print(f"[INFO] final_model  = {current_model}")
    print(f"[INFO] summary_log  = {summary_log}")
    print(f"{'=' * 70}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
