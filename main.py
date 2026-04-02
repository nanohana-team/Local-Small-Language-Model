"""
LSLM main.py

[会話フロー]
  main.py がオーケストレーター。各 LLM ノードは
  「受け取って生成して返すだけ」のシンプルなサーバー。

  1. セッション開始: 初期プロンプトをランダムに選んだノードに送信
  2. 返答を受け取り、履歴に追加
  3. 次のノードをランダムに選び、最新の返答を入力として送信
  4. loops × node_count ターン繰り返す
  5. 評価 (Gemini → OpenAI フォールバック)
  6. 高スコアターンで LoRA 学習 → Node D 再起動
  7. 次セッションへ
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import random
import signal
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ─── ロギング ──────────────────────────────────────────────────────────────

def ts() -> str:
    return time.strftime("%H:%M:%S")

def log(tag: str, msg: str) -> None:
    print(f"[{ts()}][{tag:<5}] {msg}", flush=True)


# ─── ノード設定 ──────────────────────────────────────────────────────────────

@dataclass
class NodeConfig:
    name: str
    model: str
    listen_port: int
    quantization: str = "8bit"


def make_nodes(
    learning_model: str = "google/gemma-3-270m-it",
    learning_quant: str = "4bit",
) -> List[NodeConfig]:
    """
    4 ノードを定義する。ノード間の転送は main.py が制御するため
    next_ports の概念はない。
    """
    return [
        NodeConfig("A", "microsoft/Phi-3.5-mini-instruct", 3000, "8bit"),
        NodeConfig("B", "LiquidAI/LFM2.5-1.2B-Instruct",  3001, "8bit"),
        NodeConfig("C", "Qwen/Qwen2.5-1.5B-Instruct",      3002, "8bit"),
        NodeConfig("D", learning_model,                     3003, learning_quant),
    ]


# ─── プロセス管理 ──────────────────────────────────────────────────────────────

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


# ─── .env 読み込み ──────────────────────────────────────────────────────────────

def _load_dotenv(path: Path) -> None:
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:
        pass


# ─── ユーティリティ ──────────────────────────────────────────────────────────────

def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ─── ノード呼び出し ──────────────────────────────────────────────────────────────

def call_node(
    host: str,
    port: int,
    session_id: str,
    input_text: str,
    history: List[Dict[str, Any]],
    timeout: float = 120.0,
) -> str:
    """
    1 ターン分のリクエストをノードに送り、返答テキストを返す。

    プロトコル:
      送信: {"session_id", "input", "history": [...]}
      受信: {"reply": "...", "session_id": "..."}

    クライアントは sendall 後に shutdown(SHUT_WR) して送信完了を通知する。
    サーバーは EOF を受け取ってから生成・応答する。
    """
    payload = json.dumps(
        {"session_id": session_id, "input": input_text, "history": history},
        ensure_ascii=False,
    ).encode("utf-8")

    sock = socket.create_connection((host, port), timeout=10.0)
    sock.settimeout(timeout)          # 生成待ちは長めに
    try:
        sock.sendall(payload)
        sock.shutdown(socket.SHUT_WR) # 送信完了を通知（EOF）

        chunks: List[bytes] = []
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        sock.close()

    data = b"".join(chunks)
    if not data:
        raise RuntimeError(f"port:{port} からの応答が空")

    result = json.loads(data.decode("utf-8", errors="ignore"))
    if "error" in result:
        raise RuntimeError(f"port:{port} エラー: {result['error']}")
    return str(result.get("reply", "")).strip()


# ─── 会話セッション ──────────────────────────────────────────────────────────────

def run_conversation_session(
    nodes: List[NodeConfig],
    session_id: str,
    initial_input: str,
    total_turns: int,
    sessions_dir: Path,
    call_timeout: float = 120.0,
) -> Dict[str, Any]:
    """
    main.py がオーケストレーションする会話セッション。

    各ターン:
      - 全ノードからランダムに1つ選択
      - 直前返答 (または initial_input) を入力として送信
      - 返答を履歴に追加

    終了後、最終セッション JSON を sessions_dir に書き込む。
    """
    history: List[Dict[str, Any]] = []
    created_at = time.strftime("%Y-%m-%d %H:%M:%S")

    for turn_idx in range(total_turns):
        node         = random.choice(nodes)
        current_input = history[-1]["text"] if history else initial_input

        log("CONV", (
            f"Turn {turn_idx + 1:>3}/{total_turns}  "
            f"Node {node.name}(:{node.listen_port})  "
            f"input={current_input[:35]!r}"
        ))

        try:
            reply = call_node(
                "127.0.0.1", node.listen_port,
                session_id, current_input, history,
                call_timeout,
            )
        except Exception as exc:
            log("WARN", f"Turn {turn_idx + 1} 失敗 (Node {node.name}): {exc}")
            continue  # 失敗ターンはスキップして次へ

        if not reply:
            log("WARN", f"Turn {turn_idx + 1} 空返答 (Node {node.name}) — スキップ")
            continue

        history.append({
            "model":   node.model,
            "port":    node.listen_port,
            "speaker": node.name,
            "text":    reply,
        })
        log("CONV", f"  → {reply[:70]!r}")

    # main.py が最終セッションファイルを書き込む（評価・学習が読む）
    session_data: Dict[str, Any] = {
        "session_id":   session_id,
        "input":        initial_input,
        "status":       "final",
        "turn_count":   len(history),
        "history":      history,
        "created_at":   created_at,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    session_path = sessions_dir / f"{session_id}.json"
    session_path.write_text(json.dumps(session_data, ensure_ascii=False, indent=2), encoding="utf-8")
    log("CONV", f"セッション保存  turns={len(history)}  file={session_path.name}")

    return session_data


# ─── セッション評価 ──────────────────────────────────────────────────────────────

def _build_eval_prompt(session: Dict[str, Any], learning_model: str) -> str:
    history = session.get("history", [])
    learning_indices = [i for i, t in enumerate(history) if t.get("model") == learning_model]
    all_models = sorted({str(t.get("model", "")) for t in history if t.get("model")})

    lines = [
        "会話ログを評価してください。JSONのみ出力。説明・コメント不要。",
        "",
        f"learning_model  : {learning_model}",
        f"total_turns     : {len(history)}",
        f"learning_indices: {learning_indices}",
        f"all_models      : {all_models}",
        "",
        "【出力内容】",
        "1. model_ranking  : 全モデルを overall 降順でリスト",
        "2. model_scores   : 各モデル集計スコア (overall/naturalness/consistency/informativeness/conciseness/safety)",
        "3. turn_scores    : learning_model の各ターン個別スコア",
        "   verdict: good_for_sft(>=0.75) / usable(>=0.5) / bad(<0.5) / unsafe",
        "",
        "【会話履歴（最大 40 ターン、番号は実際のインデックス）】",
    ]
    offset = max(0, len(history) - 40)
    for i, turn in enumerate(history[offset:], start=offset):
        model_short  = str(turn.get("model", "?")).split("/")[-1][:18]
        speaker      = str(turn.get("speaker", "?"))
        text_preview = str(turn.get("text", "")).replace("\n", " ")[:100]
        mark = " ★LEARNING★" if i in learning_indices else ""
        lines.append(f"[{i}]{mark} Node{speaker}({model_short}): {text_preview}")

    example = {
        "model_ranking": ["model_a", "model_b"],
        "model_scores": [
            {"model": "model_a", "overall": 0.0, "naturalness": 0.0,
             "consistency": 0.0, "informativeness": 0.0, "conciseness": 0.0, "safety": 0.0},
        ],
        "turn_scores": [
            {"turn_index": 0, "overall": 0.0, "naturalness": 0.0,
             "consistency": 0.0, "informativeness": 0.0, "conciseness": 0.0, "safety": 0.0,
             "verdict": "usable", "reason_short": ""},
        ],
    }
    lines += ["", "【出力形式】", json.dumps(example, ensure_ascii=False)]
    return "\n".join(lines)


def _is_quota_or_token_error(msg: str) -> bool:
    lower = msg.lower()
    return any(k in lower for k in [
        "quota", "429", "rate_limit", "rate limit",
        "resource_exhausted", "resourceexhausted",
        "too many token", "context length", "maximum context",
        "token limit", "context window", "input too long", "payload too large",
    ])


def _call_gemini(prompt: str, api_key: str, model: str) -> Tuple[Dict[str, Any], str]:
    try:
        from google import genai       # type: ignore
        from google.genai import types # type: ignore
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )
        text = response.text
        if not text:
            return {}, "empty response"
        return json.loads(text), ""
    except Exception as exc:
        return {}, str(exc)


def _call_openai(prompt: str, api_key: str, api_base: str, model: str) -> Tuple[Dict[str, Any], str]:
    try:
        import requests  # type: ignore
        resp = requests.post(
            f"{api_base.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2},
            timeout=120,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        content = content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        return json.loads(content), ""
    except Exception as exc:
        return {}, str(exc)


def run_session_eval(
    session_path: Path,
    eval_dir: Path,
    learning_model: str,
    gemini_key: Optional[str],
    gemini_model: str,
    fallback_key: Optional[str],
    fallback_api_base: str,
    fallback_model: str,
) -> Optional[Dict[str, Any]]:
    """
    セッションを評価し eval_dir に保存する。
    Gemini → OpenAI フォールバック順。
    戻り値: 評価 dict (turn_scores 含む) / 失敗時 None。
    """
    if not gemini_key and not fallback_key:
        log("EVAL", "API キーなし — 評価スキップ")
        return None

    try:
        session = json.loads(session_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log("EVAL", f"セッション読み込み失敗: {exc}")
        return None

    log("EVAL", f"評価開始  turns={session.get('turn_count', '?')}  model={gemini_model}")
    prompt = _build_eval_prompt(session, learning_model)

    result: Dict[str, Any] = {}
    err    = ""
    source = ""

    if gemini_key:
        result, err = _call_gemini(prompt, gemini_key, gemini_model)
        if not err:
            source = "gemini"
        else:
            log("EVAL", f"Gemini 失敗: {err[:100]}")
            if _is_quota_or_token_error(err) and fallback_key:
                log("EVAL", f"→ OpenAI フォールバック  model={fallback_model}")
                result, err = _call_openai(prompt, fallback_key, fallback_api_base, fallback_model)
                source = "openai_fallback" if not err else ""
            else:
                return None

    elif fallback_key:
        result, err = _call_openai(prompt, fallback_key, fallback_api_base, fallback_model)
        source = "openai" if not err else ""

    if err or not result:
        log("EVAL", f"評価失敗: {err[:100]}")
        return None

    result.update({
        "session_id":    session.get("session_id", session_path.stem),
        "evaluated_by":  source,
        "evaluated_at":  time.strftime("%Y-%m-%d %H:%M:%S"),
        "session_turns": session.get("turn_count", 0),
    })

    eval_dir.mkdir(parents=True, exist_ok=True)
    out = eval_dir / f"{session_path.stem}.eval.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log("EVAL", f"評価保存  source={source}  file={out.name}")

    ranking = result.get("model_ranking", [])
    if ranking:
        log("EVAL", f"ランキング: {[str(m).split('/')[-1] for m in ranking]}")
    for s in result.get("model_scores", []):
        m = str(s.get("model", "?")).split("/")[-1][:24]
        log("EVAL", f"  {m:<24}  overall={float(s.get('overall', 0.0)):.3f}")
    ts_list = result.get("turn_scores", [])
    if ts_list:
        log("EVAL", f"turn_scores: {len(ts_list)} ターン  "
            f"verdicts={[t.get('verdict','?') for t in ts_list]}")

    return result


# ─── SFT データ構築 + 学習 ──────────────────────────────────────────────

SFT_SYSTEM_PROMPT = "あなたは自然な日本語で簡潔に返答する会話AIです。"


def _build_sft_rows(
    session: Dict[str, Any],
    eval_result: Dict[str, Any],
    learning_model: str,
    min_overall: float,
) -> List[Dict[str, Any]]:
    """
    高スコアな learning ターンを SFT 形式に変換する。
    messages: [system, user(直前ターン or 初期入力), assistant(learning返答)]
    """
    history       = session.get("history", [])
    initial_input = session.get("input", "")

    ts_map: Dict[int, Dict] = {
        int(item["turn_index"]): item
        for item in eval_result.get("turn_scores", [])
        if isinstance(item.get("turn_index"), int)
    }

    rows: List[Dict[str, Any]] = []
    for i, turn in enumerate(history):
        if turn.get("model") != learning_model:
            continue
        score   = ts_map.get(i, {})
        overall = float(score.get("overall", 0.0))
        verdict = str(score.get("verdict", "bad"))
        if overall < min_overall or verdict not in ("good_for_sft", "usable"):
            continue
        text = str(turn.get("text", "")).strip()
        if not text:
            continue
        user_content = initial_input
        if i > 0:
            prev = str(history[i - 1].get("text", "")).strip()
            if prev:
                user_content = prev
        if not user_content:
            continue
        rows.append({
            "messages": [
                {"role": "system",    "content": SFT_SYSTEM_PROMPT},
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": text},
            ],
        })
    return rows


def run_training_step(
    session_path: Path,
    eval_result: Dict[str, Any],
    learning_model: str,
    train_root: Path,
    session_num: int,
    python_exe: str,
    project_root: Path,
    min_overall: float,
    eval_ratio: float,
    train_epochs: float,
    train_batch_size: int,
    eval_batch_size: int,
    grad_accum: int,
    lr: float,
    max_length: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    bf16: bool,
    fp16: bool,
    merge: bool,
) -> Optional[str]:
    """
    eval_result の turn_scores から SFT データを構築し LoRA 学習を行う。
    成功時: 新モデルパス / データなし・失敗時: None
    """
    log("TRAIN", f"学習ステップ開始  session_num={session_num}")

    try:
        session = json.loads(session_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log("TRAIN", f"セッション読み込み失敗: {exc}")
        return None

    rows = _build_sft_rows(session, eval_result, learning_model, min_overall)
    if not rows:
        log("TRAIN", f"採用ターンなし (min_overall={min_overall}) — スキップ")
        return None

    log("TRAIN", f"採用ターン: {len(rows)}")

    iter_dir = train_root / f"session_{session_num:04d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    random.shuffle(rows)
    split = max(1, int(len(rows) * (1 - eval_ratio)))
    if split >= len(rows):
        split = len(rows)

    sft_train_file = iter_dir / "sft_train.jsonl"
    sft_eval_file  = iter_dir / "sft_eval.jsonl"
    lora_out_dir   = iter_dir / "student_lora"
    merged_out_dir = iter_dir / "student_merged"

    _write_jsonl(sft_train_file, rows[:split])
    _write_jsonl(sft_eval_file,  rows[split:])
    log("TRAIN", f"データ書き出し  train={split}  eval={len(rows)-split}")

    train_cmd: List[str] = [
        python_exe,
        str(project_root / "src" / "train" / "train_gemma3_270m_sft.py"),
        "--model-name",                  learning_model,
        "--train-file",                  str(sft_train_file),
        "--eval-file",                   str(sft_eval_file),
        "--output-dir",                  str(lora_out_dir),
        "--merged-output-dir",           str(merged_out_dir),
        "--max-length",                  str(max_length),
        "--per-device-train-batch-size", str(train_batch_size),
        "--per-device-eval-batch-size",  str(eval_batch_size),
        "--gradient-accumulation-steps", str(grad_accum),
        "--num-train-epochs",            str(train_epochs),
        "--learning-rate",               str(lr),
        "--lora-r",                      str(lora_r),
        "--lora-alpha",                  str(lora_alpha),
        "--lora-dropout",                str(lora_dropout),
    ]
    if bf16:  train_cmd.append("--bf16")
    if fp16:  train_cmd.append("--fp16")
    if merge: train_cmd.append("--merge-after-training")

    log("TRAIN", f"LoRA 学習開始  base={str(learning_model).split('/')[-1]}")
    t0     = time.time()
    result = subprocess.run(train_cmd, cwd=str(project_root), check=False)
    elapsed = round(time.time() - t0, 1)

    if result.returncode != 0:
        log("TRAIN", f"学習失敗 (exit={result.returncode})  elapsed={elapsed}s")
        return None

    log("TRAIN", f"学習完了  elapsed={elapsed}s")

    if merge and merged_out_dir.exists() and any(merged_out_dir.iterdir()):
        new_model = str(merged_out_dir)
    elif lora_out_dir.exists() and any(lora_out_dir.iterdir()):
        new_model = str(lora_out_dir)
    else:
        log("TRAIN", "出力ディレクトリが空")
        return None

    log("TRAIN", f"新モデル: {new_model}")
    return new_model


# ─── Node D 再起動 ──────────────────────────────────────────────

def restart_node_d(
    node_d: NodeConfig,
    new_model: str,
    python_exe: str,
    llm_script: Path,
    device: str,
    max_new_tokens: int,
    project_root: Path,
    launcher_dir: Path,
    old_proc: Optional[subprocess.Popen],
    restart_num: int,
    startup_timeout: float,
    poll_interval: float,
) -> Tuple[NodeConfig, subprocess.Popen]:
    log("NODE", f"Node D 停止中 (restart #{restart_num})...")
    terminate_process_tree(old_proc)
    time.sleep(3.0)

    new_node = NodeConfig(node_d.name, new_model, node_d.listen_port, node_d.quantization)
    cmd      = build_node_command(python_exe, llm_script, new_node, device, max_new_tokens)
    log_file = launcher_dir / f"server_D_port{node_d.listen_port}_r{restart_num}.log"

    log("NODE", f"Node D 再起動  model={str(new_model).split('/')[-1]}")
    new_proc = launch_in_new_console(cmd, project_root, log_file)
    RUNNER.add(new_proc)

    log("NODE", f"pid={new_proc.pid}  port={node_d.listen_port} 待機中...")
    wait_for_ports("127.0.0.1", [node_d.listen_port], startup_timeout, poll_interval)
    log("NODE", "Node D 準備完了")
    return new_node, new_proc


# ─── 引数 ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "ローカル LLM リング (teacher×3 + learning×1):\n"
            "main.py がランダム順で各ノードを呼び出して会話を進める。"
        )
    )
    # 基本
    p.add_argument("text", nargs="?", default="こんにちは。今日の話題を決めてください。")
    p.add_argument("--python",           default=sys.executable)
    p.add_argument("--device",           default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--max-new-tokens",   type=int,   default=128)
    p.add_argument("--loops",            type=int,   default=5,
                   help="1セッションのラウンド数 (総ターン = loops × node数)")
    p.add_argument("--session-count",    type=int,   default=0,
                   help="セッション総数 (0=無限)")
    p.add_argument("--session-interval", type=float, default=3.0)
    p.add_argument("--startup-timeout",  type=float, default=240.0)
    p.add_argument("--poll-interval",    type=float, default=0.5)
    p.add_argument("--call-timeout",     type=float, default=120.0,
                   help="ノード生成待ちタイムアウト (秒)")
    p.add_argument("--project-root",     default=".")
    p.add_argument("--keep-running",     action="store_true")
    p.add_argument("--combine-logs",     action="store_true")

    # learning モデル (Node D)
    p.add_argument("--learning-model",   default="google/gemma-3-270m-it",
                   help="Node D のモデル (HuggingFace ID またはローカルパス)")
    p.add_argument("--learning-quant",   default="4bit", choices=["none", "4bit", "8bit"])

    # 評価 API
    p.add_argument("--env-file",                 default=".env")
    p.add_argument("--eval-gemini-model",        default="gemini-3.1-flash-lite-preview")
    p.add_argument("--eval-fallback-api-base",   default="https://api.openai.com/v1")
    p.add_argument("--eval-fallback-model",      default="gpt-4o-mini")
    p.add_argument("--eval-fallback-key-env",    default="OPENAI_API_KEY")
    p.add_argument("--skip-eval",                action="store_true")

    # 学習
    p.add_argument("--skip-train",               action="store_true")
    p.add_argument("--train-output-root",        default="runs/online_sft")
    p.add_argument("--train-min-overall",        type=float, default=0.70)
    p.add_argument("--train-eval-ratio",         type=float, default=0.05)
    p.add_argument("--train-epochs",             type=float, default=1.0)
    p.add_argument("--train-batch-size",         type=int,   default=4)
    p.add_argument("--train-eval-batch-size",    type=int,   default=4)
    p.add_argument("--train-grad-accum",         type=int,   default=4)
    p.add_argument("--train-lr",                 type=float, default=2e-4)
    p.add_argument("--train-max-length",         type=int,   default=512)
    p.add_argument("--train-lora-r",             type=int,   default=16)
    p.add_argument("--train-lora-alpha",         type=int,   default=32)
    p.add_argument("--train-lora-dropout",       type=float, default=0.05)
    p.add_argument("--train-bf16",               action="store_true")
    p.add_argument("--train-fp16",               action="store_true")
    p.add_argument("--train-merge",              action="store_true")
    return p.parse_args()


# ─── パス解決 ──────────────────────────────────────────────────────────────

def resolve_project_root(start: str) -> Path:
    candidate = Path(start).resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for root in [candidate, Path.cwd().resolve()]:
        for p in [root, *root.parents]:
            if (p / "config").exists() or (p / "src").exists() or (p / "llm.py").exists():
                return p
    return candidate


def resolve_script(project_root: Path, names: Sequence[str]) -> Path:
    for name in names:
        path = project_root / name
        if path.exists():
            return path
    raise FileNotFoundError("script not found: " + ", ".join(str(project_root / x) for x in names))


def resolve_llm_script(project_root: Path) -> Path:
    return resolve_script(project_root, [
        "llm.py", "llm/llm.py", "src/chat/llm.py", "src/chat/llm/llm.py", "src/llm.py",
    ])


def resolve_logs_dir(project_root: Path) -> Path:
    candidates = [project_root / "logs", project_root / "src" / "chat" / "logs",
                  project_root / "src" / "logs"]
    for path in candidates:
        if path.exists():
            path.mkdir(parents=True, exist_ok=True)
            return path
    candidates[0].mkdir(parents=True, exist_ok=True)
    return candidates[0]


# ─── コマンド構築 ──────────────────────────────────────────────────────────────

def build_node_command(
    python_exe: str, llm_script: Path, node: NodeConfig, device: str, max_new_tokens: int
) -> List[str]:
    """
    LLM ノード起動コマンドを生成する。
    main.py がルーティングを担うため --next-ports は不要。
    """
    return [
        python_exe, "-u", str(llm_script),
        "--model",          node.model,
        "--listen-port",    str(node.listen_port),
        "--max-new-tokens", str(max_new_tokens),
        "--device",         device,
        "--quantization",   node.quantization,
    ]


# ─── ネットワーク ──────────────────────────────────────────────────────────────

def wait_for_ports(host: str, ports: Sequence[int], timeout: float, interval: float) -> None:
    log("WAIT", f"ポート待機中 {list(ports)}  timeout={timeout:.0f}s")
    start    = time.time()
    last_log = start
    while True:
        if all(_try_connect(host, p) for p in ports):
            log("WAIT", f"全ポート疎通確認  elapsed={time.time()-start:.1f}s")
            return
        if time.time() - start > timeout:
            raise TimeoutError(f"Timeout waiting for ports: {list(ports)}")
        now = time.time()
        if now - last_log >= 15.0:
            log("WAIT", f"待機継続... {now-start:.0f}s / {timeout:.0f}s")
            last_log = now
        time.sleep(interval)


def _try_connect(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except Exception:
        return False


# ─── セッションサマリー表示 ──────────────────────────────────────────────

def print_session_summary(session_data: Dict[str, Any], session_num: int) -> None:
    history = session_data.get("history", [])
    print(flush=True)
    print(f"[{ts()}] {'━'*60}", flush=True)
    print(f"[{ts()}]   SESSION #{session_num} 完了", flush=True)
    print(f"[{ts()}]   session_id : {str(session_data.get('session_id',''))[:8]}...", flush=True)
    print(f"[{ts()}]   turns      : {session_data.get('turn_count', len(history))}", flush=True)
    if history:
        last        = history[-1]
        model_short = str(last.get("model", "?")).split("/")[-1][:24]
        speaker     = str(last.get("speaker", "?"))
        preview     = str(last.get("text", ""))[:80].replace("\n", " ")
        print(f"[{ts()}]   last_speaker: Node {speaker} ({model_short})", flush=True)
        print(f"[{ts()}]   last_reply  : {preview}", flush=True)
    print(f"[{ts()}] {'━'*60}", flush=True)


# ─── プロセス終了 ──────────────────────────────────────────────────────────────

def kill_process_by_port(port: int) -> None:
    if os.name != "nt":
        return
    try:
        result = subprocess.check_output(
            f'netstat -ano | findstr :{port}', shell=True, encoding='utf-8'
        )
        for line in result.strip().splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[-1].isdigit():
                subprocess.run(["taskkill", "/PID", parts[-1], "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        pass


def terminate_process_tree(proc: "subprocess.Popen | None") -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        pass


def launch_in_new_console(cmd: List[str], cwd: Path, log_file: Path) -> subprocess.Popen:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        quoted = subprocess.list2cmdline(cmd)
        return subprocess.Popen(f'cmd.exe /k {quoted}', cwd=str(cwd),
                                creationflags=subprocess.CREATE_NEW_CONSOLE)
    with log_file.open("a", encoding="utf-8") as fh:
        return subprocess.Popen(cmd, cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT)


# ─── メイン ──────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    project_root = resolve_project_root(args.project_root)
    llm_script   = resolve_llm_script(project_root)
    logs_dir     = resolve_logs_dir(project_root)
    sessions_dir = logs_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    eval_dir   = logs_dir / "evals"
    train_root = (project_root / args.train_output_root).resolve()

    _load_dotenv(project_root / args.env_file)
    gemini_key   = os.environ.get("GEMINI_API_KEY")
    fallback_key = os.environ.get(args.eval_fallback_key_env)

    nodes      = make_nodes(args.learning_model, args.learning_quant)
    node_d_cfg = nodes[3]
    all_ports  = [n.listen_port for n in nodes]
    total_turns_per_session = args.loops * len(nodes)

    current_learning_model = args.learning_model

    run_id       = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    launcher_dir = logs_dir / "launcher" / run_id
    launcher_dir.mkdir(parents=True, exist_ok=True)
    loop_label   = "∞" if args.session_count == 0 else str(args.session_count)

    (launcher_dir / "run_manifest.json").write_text(
        json.dumps({
            "run_id": run_id, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "nodes": [asdict(n) for n in nodes],
            "total_turns_per_session": total_turns_per_session,
            "session_count": args.session_count,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    atexit.register(RUNNER.stop_all)

    print(f"\n[{ts()}] {'='*65}", flush=True)
    log("INFO", f"run_id         : {run_id}")
    log("INFO", f"ノード構成     :")
    for n in nodes:
        tag = " ← learning (Node D)" if n.name == "D" else ""
        log("INFO", f"  Node {n.name} port={n.listen_port}  {str(n.model).split('/')[-1]}  quant={n.quantization}{tag}")
    log("INFO", f"発言順         : ランダム（ターンごとに全ノードから抽選）")
    log("INFO", f"総ターン/session: {total_turns_per_session} ({args.loops} loops × {len(nodes)} nodes)")
    log("INFO", f"session_count  : {loop_label}")
    log("INFO", f"eval           : {'有効' if (gemini_key or fallback_key) and not args.skip_eval else '無効'}")
    log("INFO", f"train          : {'有効' if not args.skip_train else '無効'}")
    print(f"[{ts()}] {'='*65}", flush=True)

    if args.combine_logs:
        combine_script = resolve_script(project_root, [
            "combine_logs.py", "src/chat/combine_logs.py", "src/combine_logs.py",
        ])
        proc = launch_in_new_console([args.python, "-u", str(combine_script)],
                                     project_root, launcher_dir / "combine_logs.log")
        RUNNER.add(proc)
        log("INFO", f"combine_logs 起動 pid={proc.pid}")

    # LLM ノード起動
    log("INFO", f"LLM ノードを {len(nodes)} 台起動中...")
    node_d_proc: Optional[subprocess.Popen] = None
    for node in nodes:
        cmd      = build_node_command(args.python, llm_script, node, args.device, args.max_new_tokens)
        log_file = launcher_dir / f"server_{node.name}_port{node.listen_port}.log"
        proc     = launch_in_new_console(cmd, project_root, log_file)
        RUNNER.add(proc)
        log("INFO", f"  Node {node.name} 起動  port={node.listen_port}  pid={proc.pid}")
        if node.name == "D":
            node_d_proc = proc
        time.sleep(1.0)

    wait_for_ports("127.0.0.1", all_ports, args.startup_timeout, args.poll_interval)

    # ━━━ セッションループ ━━━
    session_num   = 0
    d_restart_num = 0
    log("INFO", "セッションループ開始  (Ctrl+C で停止)")

    try:
        while True:
            session_num += 1
            print(f"\n[{ts()}] {'─'*65}", flush=True)
            log("LOOP", (
                f"セッション #{session_num}/{loop_label}  "
                f"learning={str(current_learning_model).split('/')[-1]}"
            ))

            session_id   = str(uuid.uuid4())
            session_data = run_conversation_session(
                nodes, session_id, args.text,
                total_turns_per_session, sessions_dir,
                args.call_timeout,
            )
            session_path = sessions_dir / f"{session_id}.json"

            print_session_summary(session_data, session_num)
            (launcher_dir / "last_session_path.txt").write_text(str(session_path), encoding="utf-8")

            # 評価
            eval_result: Optional[Dict[str, Any]] = None
            if not args.skip_eval:
                eval_result = run_session_eval(
                    session_path, eval_dir, current_learning_model,
                    gemini_key, args.eval_gemini_model,
                    fallback_key, args.eval_fallback_api_base, args.eval_fallback_model,
                )

            # 学習 + Node D 再起動
            if eval_result and not args.skip_train:
                new_model = run_training_step(
                    session_path, eval_result, current_learning_model,
                    train_root, session_num, args.python, project_root,
                    args.train_min_overall, args.train_eval_ratio,
                    args.train_epochs, args.train_batch_size, args.train_eval_batch_size,
                    args.train_grad_accum, args.train_lr, args.train_max_length,
                    args.train_lora_r, args.train_lora_alpha, args.train_lora_dropout,
                    args.train_bf16, args.train_fp16, args.train_merge,
                )
                if new_model:
                    d_restart_num += 1
                    node_d_cfg, node_d_proc = restart_node_d(
                        node_d_cfg, new_model,
                        args.python, llm_script, args.device, args.max_new_tokens,
                        project_root, launcher_dir, node_d_proc,
                        d_restart_num, args.startup_timeout, args.poll_interval,
                    )
                    current_learning_model = new_model

            # 終了判定
            if args.session_count > 0 and session_num >= args.session_count:
                log("LOOP", f"指定セッション数 ({args.session_count}) に達したため終了")
                break

            log("LOOP", f"次のセッションまで {args.session_interval}s 待機...")
            time.sleep(args.session_interval)

    except KeyboardInterrupt:
        print(flush=True)
        log("INFO", "Ctrl+C — ループ停止")

    if not args.keep_running:
        log("INFO", "LLM サーバーを停止中...")
        RUNNER.stop_all()
        for port in all_ports:
            kill_process_by_port(port)
        log("INFO", "全プロセス停止完了")
    else:
        log("INFO", "--keep-running: サーバー稼働継続")

    log("INFO", f"合計 session={session_num}  Node D 再起動={d_restart_num}回")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
