import argparse
import json
import random
import socket
import sys
import time
import uuid
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig


# ---------------------------------------------------------------------------
# ロギングユーティリティ
# ---------------------------------------------------------------------------

def ts() -> str:
    """現在時刻 HH:MM:SS"""
    return time.strftime("%H:%M:%S")


def utc_timestamp() -> str:
    now = time.time()
    base = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
    ms = int((now - int(now)) * 1000)
    return f"{base}.{ms:03d}"


def plog(port: int, tag: str, msg: str) -> None:
    """ポート番号付きのコンソールログ"""
    print(f"[{ts()}][:{port}][{tag:<5}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# モデルロード
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_name: str, device: str, quantization: str
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, torch.device]:

    print(f"[{ts()}][LOAD ] トークナイザ読み込み: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    model_kwargs: Dict[str, Any] = {}
    device_obj = torch.device("cpu")

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda が指定されましたが CUDA が使用できません。")
        device_obj = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[{ts()}][LOAD ] GPU: {gpu_name}  VRAM: {vram_gb:.1f}GB", flush=True)

        if quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["device_map"] = "auto"
            print(f"[{ts()}][LOAD ] 4bit 量子化で読み込み", flush=True)
        elif quantization == "8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["device_map"] = "auto"
            print(f"[{ts()}][LOAD ] 8bit 量子化で読み込み", flush=True)
        else:
            model_kwargs["torch_dtype"] = torch.float16
            print(f"[{ts()}][LOAD ] fp16 で読み込み", flush=True)
    else:
        if quantization != "none":
            print(f"[{ts()}][WARN ] quantization={quantization} は device=cpu では無効", flush=True)
        quantization = "none"
        print(f"[{ts()}][LOAD ] CPU で読み込み (float32)", flush=True)

    t0 = time.time()
    print(f"[{ts()}][LOAD ] モデル読み込み開始... (時間がかかる場合があります)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    elapsed = time.time() - t0
    print(f"[{ts()}][LOAD ] モデル読み込み完了  {elapsed:.1f}s", flush=True)

    if device == "cuda" and quantization == "none":
        model = model.to(device_obj)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            model.resize_token_embeddings(len(tokenizer))

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_length = None
        model.generation_config.max_new_tokens = None

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[{ts()}][LOAD ] パラメータ数: {param_count:.1f}M", flush=True)

    return model, tokenizer, device_obj


# ---------------------------------------------------------------------------
# LocalLLMBridge
# ---------------------------------------------------------------------------

class LocalLLMBridge:
    def __init__(
        self,
        model_name: str,
        listen_host: str,
        listen_port: int,
        next_ports: List[int],
        max_new_tokens: int = 128,
        device: str = "cpu",
        quantization: str = "none",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_history_turns: int = 12,
        history_char_budget: int = 4000,
        persist_history_items: int = 2000,
        system_prompt: str = "あなたは自然な日本語で簡潔に返答する会話AIです。",
    ) -> None:
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.next_ports = next_ports
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.model_name = model_name
        self.quantization = quantization
        self.temperature = temperature
        self.top_p = top_p
        self.max_history_turns = max_history_turns
        self.history_char_budget = history_char_budget
        self.persist_history_items = persist_history_items
        self.system_prompt = system_prompt
        self.pid = os.getpid()

        # ログディレクトリ: src/chat/llm.py → parents[1]=src → parents[2]=project_root
        script_dir   = Path(__file__).resolve().parent
        project_root = script_dir.parents[1]
        self.logs_dir    = project_root / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir = self.logs_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.logs_dir / f"node_port{listen_port}.jsonl"

        # 起動ログ
        print(f"\n[{ts()}] {'='*55}", flush=True)
        plog(listen_port, "INIT", f"モデル        : {model_name}")
        plog(listen_port, "INIT", f"デバイス      : {device} / quant={quantization}")
        plog(listen_port, "INIT", f"最大トークン  : {max_new_tokens}")
        plog(listen_port, "INIT", f"次ポート      : {next_ports}")
        plog(listen_port, "INIT", f"ログ          : {self.log_path}")
        plog(listen_port, "INIT", f"セッション保存: {self.sessions_dir}")
        plog(listen_port, "INIT", f"PID           : {self.pid}")
        print(f"[{ts()}] {'='*55}", flush=True)

        self.model, self.tokenizer, self.device_obj = load_model_and_tokenizer(
            model_name, device, quantization
        )
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature > 0),
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        plog(listen_port, "READY", "モデルロード完了 — リクエスト待機中")

    # ─── ログ書き込み ──────────────────────────────────────────────

    def write_log(self, event: Dict[str, Any]) -> None:
        event = dict(event)
        event["timestamp"] = utc_timestamp()
        event["pid"] = self.pid
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
            f.flush()

    # ─── セッション管理 ──────────────────────────────────────────────

    def get_session_path(self, session_id: str) -> Path:
        safe = "".join(c for c in session_id if c.isalnum() or c in ("-", "_"))
        return self.sessions_dir / f"{safe}.json"

    def load_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        path = self.get_session_path(session_id)
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("history"), list):
                return data["history"]
        except Exception as exc:
            self.write_log({"event": "history_load_error", "session_id": session_id,
                            "port": self.listen_port, "error": str(exc)})
        return []

    def save_session_history(
        self,
        session_id: str,
        history: List[Dict[str, Any]],
        status: str = "running",
        input_text: str = "",
    ) -> None:
        """
        セッションファイルを書き込む。
        status:
          "running" … 会話継続中
          "final"   … TTL=0 で正常完了
          "end"     … TTL=0 受信・生成なし終了
        input_text: 最初に送られた入力テキスト（SFT データ構築に使用）
        """
        path = self.get_session_path(session_id)
        trimmed = history[-self.persist_history_items:] if self.persist_history_items > 0 else history
        payload = {
            "session_id": session_id,
            "model":      self.model_name,
            "updated_at": utc_timestamp(),
            "status":     status,
            "turn_count": len(trimmed),
            "input":      input_text,   # SFT 構築時にユーザー入力として使用
            "history":    trimmed,
        }
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp.replace(path)

    def merge_histories(
        self,
        stored: List[Dict[str, Any]],
        incoming: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen = set()
        for item in stored + incoming:
            model = str(item.get("model", "")).strip()
            port  = item.get("port")
            text  = str(item.get("text", "")).strip()
            if not text:
                continue
            key = (model, port, text)
            if key in seen:
                continue
            seen.add(key)
            merged.append({"model": model, "port": port, "text": text})
        if self.persist_history_items > 0:
            merged = merged[-self.persist_history_items:]
        return merged

    # ─── プロンプト構築 ──────────────────────────────────────────────

    def build_prompt(self, input_text: str, history: List[Dict[str, Any]]) -> str:
        selected = history[-self.max_history_turns:] if self.max_history_turns > 0 else []
        rendered: List[str] = []
        total = 0
        for item in reversed(selected):
            speaker = item.get("model") or f"port_{item.get('port','?')}"
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            line = f"{speaker}: {text}"
            if total + len(line) > self.history_char_budget:
                break
            rendered.append(line)
            total += len(line)
        rendered.reverse()
        block = "\n".join(rendered).strip() or "（履歴なし）"
        return f"{self.system_prompt}\n\n以下はこれまでの会話履歴です。\n{block}\n\nユーザー: {input_text}\nAI:"

    # ─── 返答クリーニング ──────────────────────────────────────────────

    def clean_reply(self, reply: str) -> str:
        reply = reply.strip()
        if not reply:
            return "了解です。"
        for marker in ["\nユーザー:", "\nAI:", "\nHuman:", "\nUser:",
                       "\nAssistant:", "\nassistant:", "\nuser:", "\nmodel="]:
            idx = reply.find(marker)
            if idx != -1:
                reply = reply[:idx].strip()
        lines = [ln.strip() for ln in reply.splitlines() if ln.strip()]
        if not lines:
            return "了解です。"
        cleaned = "\n".join(lines).strip()
        changed = True
        while changed:
            changed = False
            for prefix in ("AI:", "Assistant:", "assistant:", "回答:", "応答:"):
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
                    changed = True
        return cleaned or "了解です。"

    # ─── 生成 ──────────────────────────────────────────────────────

    def generate_reply(self, input_text: str, history: List[Dict[str, Any]]) -> str:
        prompt = self.build_prompt(input_text, history)
        plog(self.listen_port, "GEN", f"生成開始  prompt={len(prompt)}chars  history={len(history)}turns")
        t0 = time.perf_counter()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device_obj) for k, v in inputs.items()}
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, generation_config=self.generation_config)
        elapsed = time.perf_counter() - t0
        prompt_len = inputs["input_ids"].shape[1]
        new_token_ids = output_ids[0][prompt_len:]
        cleaned = self.clean_reply(self.tokenizer.decode(new_token_ids, skip_special_tokens=True))
        new_tokens = len(new_token_ids)
        tok_per_sec = new_tokens / elapsed if elapsed > 0 else 0
        plog(self.listen_port, "GEN",
             f"生成完了  {new_tokens}tok  {elapsed:.2f}s  {tok_per_sec:.1f}tok/s  "
             f"reply={cleaned[:40]!r}")
        self.write_log({
            "event": "generate_metrics",
            "port": self.listen_port,
            "prompt_chars": len(prompt),
            "new_tokens": new_tokens,
            "reply_chars": len(cleaned),
            "elapsed_sec": round(elapsed, 4),
            "tok_per_sec": round(tok_per_sec, 2),
        })
        return cleaned

    # ─── 次ノードへ転送 ──────────────────────────────────────────────

    def send_next(self, message: Dict[str, Any]) -> Optional[int]:
        if not self.next_ports:
            return None
        port = random.choice(self.next_ports)
        payload = json.dumps(message, ensure_ascii=False).encode("utf-8")
        plog(self.listen_port, "FWD", f"-> port:{port}  ttl={message.get('ttl')}  "
             f"history={len(message.get('history', []))}turns")
        with socket.create_connection(("127.0.0.1", port), timeout=10.0) as sock:
            sock.sendall(payload)
        return port

    # ─── クライアント処理 ──────────────────────────────────────────────

    def handle_client(self, conn: socket.socket, addr: tuple) -> None:
        """
        main.py から送られた 1 ターン分のリクエストを処理する。

        プロトコル (main.py がオーケストレーター):
          受信: {"session_id": "...", "input": "...", "history": [...]}
          送信: {"reply": "...", "session_id": "..."}

        ノードは転送を行わない。次の発話者は main.py が決定する。
        """
        message: Optional[Dict[str, Any]] = None
        try:
            # クライアントが shutdown(SHUT_WR) するまで全データを受信
            chunks: List[bytes] = []
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
            data = b"".join(chunks)
            if not data:
                return

            message     = json.loads(data.decode("utf-8", errors="ignore"))
            session_id  = str(message.get("session_id") or uuid.uuid4())
            sid_short   = session_id[:8]
            input_text  = str(message.get("input", "")).strip()
            incoming_history = message.get("history", [])
            if not isinstance(incoming_history, list):
                incoming_history = []

            plog(self.listen_port, "RECV",
                 f"session={sid_short}  hist={len(incoming_history)}  input={input_text[:40]!r}")

            stored_history = self.load_session_history(session_id)
            history = self.merge_histories(stored_history, incoming_history)

            self.write_log({
                "event": "receive",
                "session_id": session_id,
                "port": self.listen_port,
                "from_addr": f"{addr[0]}:{addr[1]}",
                "input": input_text,
                "incoming_history_len": len(incoming_history),
                "stored_history_len":   len(stored_history),
                "merged_history_len":   len(history),
            })

            # 生成
            reply = self.generate_reply(input_text, history)
            history.append({"model": self.model_name, "port": self.listen_port, "text": reply})
            if self.persist_history_items > 0:
                history = history[-self.persist_history_items:]

            # 中間状態を保存（main.py がセッション完了後に最終状態を上書きする）
            self.save_session_history(session_id, history, status="running", input_text=input_text)

            self.write_log({
                "event": "reply",
                "session_id": session_id,
                "port": self.listen_port,
                "input": input_text,
                "reply": reply,
                "history_len": len(history),
            })

            # 返答を main.py に返す（転送は行わない）
            response_bytes = json.dumps(
                {"reply": reply, "session_id": session_id},
                ensure_ascii=False,
            ).encode("utf-8")
            conn.sendall(response_bytes)

        except Exception as exc:
            err_msg = f"[{ts()}][:{self.listen_port}][ERR  ] {exc}"
            print(err_msg, file=sys.stderr, flush=True)
            self.write_log({
                "event": "error",
                "session_id": message.get("session_id") if isinstance(message, dict) else None,
                "port": self.listen_port,
                "error": str(exc),
            })
            try:
                conn.sendall(json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8"))
            except Exception:
                pass

        except Exception as exc:
            err_msg = f"[{ts()}][:{self.listen_port}][ERR  ] {exc}"
            print(err_msg, file=sys.stderr, flush=True)
            self.write_log({
                "event": "error",
                "session_id": (message or {}).get("session_id") if isinstance(message, dict) else None,
                "port": self.listen_port,
                "error": str(exc),
            })
            try:
                conn.sendall(err_msg.encode("utf-8", errors="ignore"))
            except Exception:
                pass

    # ─── サーバーループ ──────────────────────────────────────────────

    def serve_forever(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.listen_host, self.listen_port))
            server.listen(5)
            plog(self.listen_port, "START",
                 f"リスニング開始  {self.listen_host}:{self.listen_port}")
            self.write_log({
                "event": "startup",
                "port": self.listen_port,
                "next_ports": self.next_ports,
                "model": self.model_name,
                "device": self.device,
                "quantization": self.quantization,
                "max_new_tokens": self.max_new_tokens,
            })
            conn_count = 0
            while True:
                conn, addr = server.accept()
                conn_count += 1
                plog(self.listen_port, "CONN",
                     f"接続受付 #{conn_count}  from {addr[0]}:{addr[1]}")
                with conn:
                    self.handle_client(conn, addr)


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",                  default="LiquidAI/LFM2.5-1.2B-Instruct")
    parser.add_argument("--listen-port",            type=int, required=True)
    parser.add_argument("--next-ports",             type=int, nargs="*", default=[])
    parser.add_argument("--max-new-tokens",         type=int, default=128)
    parser.add_argument("--device",                 default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--quantization",           default="none", choices=["none", "4bit", "8bit"])
    parser.add_argument("--temperature",            type=float, default=0.7)
    parser.add_argument("--top-p",                  type=float, default=0.9)
    parser.add_argument("--max-history-turns",      type=int, default=12)
    parser.add_argument("--history-char-budget",    type=int, default=4000)
    parser.add_argument("--persist-history-items",  type=int, default=2000)
    parser.add_argument("--system-prompt",
                        default="あなたは自然な日本語で簡潔に返答する会話AIです。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bridge = LocalLLMBridge(
        args.model, "127.0.0.1", args.listen_port, args.next_ports,
        args.max_new_tokens, args.device, args.quantization,
        args.temperature, args.top_p,
        args.max_history_turns, args.history_char_budget, args.persist_history_items,
        args.system_prompt,
    )
    bridge.serve_forever()


if __name__ == "__main__":
    main()
