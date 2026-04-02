import argparse
import json
import random
import socket
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_model_and_tokenizer(
    model_name: str,
    device: str,
    quantization: str,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, torch.device]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    model_kwargs: Dict[str, Any] = {}
    device_obj = torch.device("cpu")

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda was requested, but CUDA is not available.")
        device_obj = torch.device("cuda")

        if quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["device_map"] = "auto"
        elif quantization == "8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float16
    else:
        if quantization != "none":
            print(f"[WARN] quantization={quantization} は device=cpu では無効化します。")
        quantization = "none"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

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

    return model, tokenizer, device_obj


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

        # start_llm_server.py 側の llm/llm.py と直置き llm.py の両方で破綻しないようにする
        script_dir = Path(__file__).resolve().parent
        if script_dir.name == "llm":
            base_dir = script_dir.parent
        else:
            base_dir = script_dir

        self.logs_dir = base_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.sessions_dir = self.logs_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = self.logs_dir / f"port_{listen_port}.jsonl"

        print(f"[INFO] Loading model         : {model_name}")
        print(f"[INFO] Logs                  : {self.log_path}")
        print(f"[INFO] Sessions              : {self.sessions_dir}")
        print(f"[INFO] Device                : {device}")
        print(f"[INFO] Quantization          : {quantization}")
        print(f"[INFO] Max new tokens        : {max_new_tokens}")
        print(f"[INFO] Max history turns     : {max_history_turns}")
        print(f"[INFO] History char budget   : {history_char_budget}")
        print(f"[INFO] Persist history items : {persist_history_items}")

        self.model, self.tokenizer, self.device_obj = load_model_and_tokenizer(
            model_name=model_name,
            device=device,
            quantization=quantization,
        )

        self.generation_config = GenerationConfig(
            max_length=None,
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature > 0),
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    def write_log(self, event: Dict[str, Any]) -> None:
        event = dict(event)
        event["timestamp"] = utc_timestamp()
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
            f.flush()

    def get_session_path(self, session_id: str) -> Path:
        safe_session_id = "".join(c for c in session_id if c.isalnum() or c in ("-", "_"))
        return self.sessions_dir / f"{safe_session_id}.json"

    def load_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        session_path = self.get_session_path(session_id)
        if not session_path.exists():
            return []

        try:
            with open(session_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("history"), list):
                return data["history"]
        except Exception as exc:
            self.write_log(
                {
                    "event": "history_load_error",
                    "session_id": session_id,
                    "port": self.listen_port,
                    "error": str(exc),
                }
            )

        return []

    def save_session_history(self, session_id: str, history: List[Dict[str, Any]]) -> None:
        session_path = self.get_session_path(session_id)
        trimmed = history[-self.persist_history_items:] if self.persist_history_items > 0 else history

        payload = {
            "session_id": session_id,
            "model": self.model_name,
            "updated_at": utc_timestamp(),
            "history": trimmed,
        }

        tmp_path = session_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp_path.replace(session_path)

    def merge_histories(
        self,
        stored_history: List[Dict[str, Any]],
        incoming_history: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen = set()

        for item in stored_history + incoming_history:
            model = str(item.get("model", "")).strip()
            port = item.get("port")
            text = str(item.get("text", "")).strip()
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

    def build_prompt(self, input_text: str, history: List[Dict[str, Any]]) -> str:
        selected = history[-self.max_history_turns:] if self.max_history_turns > 0 else []

        rendered_lines: List[str] = []
        total_chars = 0

        for item in reversed(selected):
            speaker = item.get("model") or f"port_{item.get('port', 'unknown')}"
            text = str(item.get("text", "")).strip()
            if not text:
                continue

            line = f"{speaker}: {text}"
            if total_chars + len(line) > self.history_char_budget:
                break

            rendered_lines.append(line)
            total_chars += len(line)

        rendered_lines.reverse()

        history_block = "\n".join(rendered_lines).strip()
        if not history_block:
            history_block = "（履歴なし）"

        return (
            f"{self.system_prompt}\n\n"
            "以下はこれまでの会話履歴です。\n"
            f"{history_block}\n\n"
            f"ユーザー: {input_text}\n"
            "AI:"
        )

    def clean_reply(self, reply: str) -> str:
        reply = reply.strip()
        if not reply:
            return "了解です。"

        # 自分で埋め込んだ会話ラベルや次話者へのはみ出しを削る
        stop_markers = [
            "\nユーザー:",
            "\nAI:",
            "\nHuman:",
            "\nUser:",
            "\nAssistant:",
            "\nassistant:",
            "\nuser:",
            "\nmodel=",
        ]
        cut = len(reply)
        for marker in stop_markers:
            idx = reply.find(marker)
            if idx != -1:
                cut = min(cut, idx)
        reply = reply[:cut].strip()

        lines = [line.strip() for line in reply.splitlines() if line.strip()]
        if not lines:
            return "了解です。"

        cleaned = "\n".join(lines).strip()

        # 露骨な role prefix を先頭から除去
        prefixes = ("AI:", "Assistant:", "assistant:", "回答:", "応答:")
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
                    changed = True

        return cleaned or "了解です。"

    def generate_reply(self, input_text: str, history: List[Dict[str, Any]]) -> str:
        prompt = self.build_prompt(input_text=input_text, history=history)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device_obj) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
            )

        prompt_len = inputs["input_ids"].shape[1]
        new_token_ids = output_ids[0][prompt_len:]
        reply = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
        return self.clean_reply(reply)

    def send_next(self, message: Dict[str, Any]) -> Optional[int]:
        if not self.next_ports:
            return None

        port = random.choice(self.next_ports)
        payload = json.dumps(message, ensure_ascii=False).encode("utf-8")
        print(f"[SEND] -> 127.0.0.1:{port}")

        with socket.create_connection(("127.0.0.1", port), timeout=10.0) as sock:
            sock.sendall(payload)

        return port

    def handle_client(self, conn: socket.socket, addr: tuple[str, int]) -> None:
        try:
            data = conn.recv(65536)
            if not data:
                return

            message = json.loads(data.decode("utf-8", errors="ignore"))

            session_id = str(message.get("session_id") or uuid.uuid4())
            message["session_id"] = session_id

            ttl = int(message.get("ttl", 0))
            incoming_history = message.get("history", [])
            if not isinstance(incoming_history, list):
                incoming_history = []

            input_text = str(message.get("input", "")).strip()

            stored_history = self.load_session_history(session_id)
            history = self.merge_histories(stored_history, incoming_history)

            self.write_log(
                {
                    "event": "receive",
                    "session_id": session_id,
                    "port": self.listen_port,
                    "from_addr": f"{addr[0]}:{addr[1]}",
                    "ttl": ttl,
                    "input": input_text,
                    "incoming_history_len": len(incoming_history),
                    "stored_history_len": len(stored_history),
                    "merged_history_len": len(history),
                }
            )

            if ttl <= 0:
                self.write_log(
                    {
                        "event": "end",
                        "session_id": session_id,
                        "port": self.listen_port,
                        "ttl": ttl,
                        "input": input_text,
                        "history": history,
                        "history_len": len(history),
                    }
                )
                conn.sendall(b"END")
                return

            reply = self.generate_reply(input_text=input_text, history=history)

            history.append(
                {
                    "model": self.model_name,
                    "port": self.listen_port,
                    "text": reply,
                }
            )
            if self.persist_history_items > 0:
                history = history[-self.persist_history_items:]

            self.save_session_history(session_id, history)

            message["history"] = history
            message["ttl"] = ttl - 1

            print(f"[LLM ] {reply}")
            print(f"[TTL ] {message['ttl']}")
            print(f"[HIST] {len(history)} items")

            self.write_log(
                {
                    "event": "reply",
                    "session_id": session_id,
                    "port": self.listen_port,
                    "ttl": message["ttl"],
                    "input": input_text,
                    "reply": reply,
                    "history": history,
                    "history_len": len(history),
                }
            )

            if message["ttl"] <= 0:
                self.write_log(
                    {
                        "event": "final",
                        "session_id": session_id,
                        "port": self.listen_port,
                        "ttl": 0,
                        "input": input_text,
                        "history": history,
                        "history_len": len(history),
                    }
                )
                conn.sendall(b"FINAL")
                return

            next_port = self.send_next(message)

            self.write_log(
                {
                    "event": "forward",
                    "session_id": session_id,
                    "port": self.listen_port,
                    "to_port": next_port,
                    "ttl": message["ttl"],
                    "input": input_text,
                    "history": history,
                    "history_len": len(history),
                }
            )

            conn.sendall(b"OK")

        except Exception as exc:
            error_message = f"[ERROR] {exc}"
            print(error_message, file=sys.stderr)

            self.write_log(
                {
                    "event": "error",
                    "session_id": message.get("session_id") if isinstance(locals().get("message"), dict) else None,
                    "port": self.listen_port,
                    "error": str(exc),
                }
            )

            try:
                conn.sendall(error_message.encode("utf-8", errors="ignore"))
            except Exception:
                pass

    def serve_forever(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.listen_host, self.listen_port))
            server.listen(5)

            print(f"[START] Listening on {self.listen_host}:{self.listen_port}")

            self.write_log(
                {
                    "event": "startup",
                    "port": self.listen_port,
                    "next_ports": self.next_ports,
                    "model": self.model_name,
                    "device": self.device,
                    "quantization": self.quantization,
                    "max_new_tokens": self.max_new_tokens,
                    "max_history_turns": self.max_history_turns,
                    "history_char_budget": self.history_char_budget,
                    "persist_history_items": self.persist_history_items,
                }
            )

            while True:
                conn, addr = server.accept()
                with conn:
                    self.handle_client(conn, addr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LiquidAI/LFM2.5-1.2B-Instruct")
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--next-ports", type=int, nargs="*", default=[])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--quantization", default="none", choices=["none", "4bit", "8bit"])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-history-turns", type=int, default=12)
    parser.add_argument("--history-char-budget", type=int, default=4000)
    parser.add_argument("--persist-history-items", type=int, default=2000)
    parser.add_argument(
        "--system-prompt",
        default="あなたは自然な日本語で簡潔に返答する会話AIです。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bridge = LocalLLMBridge(
        model_name=args.model,
        listen_host="127.0.0.1",
        listen_port=args.listen_port,
        next_ports=args.next_ports,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        quantization=args.quantization,
        temperature=args.temperature,
        top_p=args.top_p,
        max_history_turns=args.max_history_turns,
        history_char_budget=args.history_char_budget,
        persist_history_items=args.persist_history_items,
        system_prompt=args.system_prompt,
    )
    bridge.serve_forever()


if __name__ == "__main__":
    main()
