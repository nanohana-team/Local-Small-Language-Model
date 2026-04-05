from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
except Exception:
    FastAPI = None
    HTTPException = Exception
    JSONResponse = None
    uvicorn = None


# =========================================================
# utils
# =========================================================


def log(message: str) -> None:
    print(message, flush=True)


def cuda_available() -> bool:
    return torch.cuda.is_available()


def pick_compute_dtype() -> torch.dtype:
    if not cuda_available():
        return torch.float16

    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def device_name() -> str:
    if not cuda_available():
        return "cpu"
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return "cuda"


def now_unix() -> int:
    return int(time.time())


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def trim_stop_text(text: str, stop: Optional[List[str]]) -> str:
    if not text or not stop:
        return text

    best_cut: Optional[int] = None
    for s in stop:
        if not s:
            continue
        idx = text.find(s)
        if idx >= 0:
            if best_cut is None or idx < best_cut:
                best_cut = idx

    if best_cut is None:
        return text
    return text[:best_cut]


# =========================================================
# config
# =========================================================


@dataclass
class LocalLLMConfig:
    model_name: str
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    do_sample: bool = True
    host: str = "127.0.0.1"
    port: int = 8000
    trust_remote_code: bool = True
    load_in_4bit: bool = True
    use_double_quant: bool = True
    quant_type: str = "nf4"
    history_char_budget: int = 6000


# =========================================================
# model wrapper
# =========================================================


class LocalQwen3Engine:
    def __init__(self, config: LocalLLMConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.model_device = "cpu"
        self.compute_dtype = pick_compute_dtype()

    def load(self) -> None:
        log(f"[LOCAL LLM] loading model={self.config.model_name}")
        log(f"[LOCAL LLM] cuda={'yes' if cuda_available() else 'no'} device={device_name()}")
        log(f"[LOCAL LLM] compute_dtype={self.compute_dtype}")

        if self.config.load_in_4bit and not cuda_available():
            raise RuntimeError(
                "4bit quantization requires CUDA. GPU was not detected."
            )

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        quant_config = None
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
        }

        if self.config.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=self.config.use_double_quant,
                bnb_4bit_quant_type=self.config.quant_type,
                bnb_4bit_compute_dtype=self.compute_dtype,
            )
            model_kwargs["quantization_config"] = quant_config
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = self.compute_dtype
            model_kwargs["device_map"] = "auto" if cuda_available() else None

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )
        model.eval()

        self.tokenizer = tokenizer
        self.model = model

        try:
            self.model_device = str(next(model.parameters()).device)
        except Exception:
            self.model_device = "unknown"

        log(f"[LOCAL LLM] model ready device={self.model_device}")

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        assert self.tokenizer is not None

        cleaned_messages: List[Dict[str, str]] = []
        for msg in messages:
            role = str(msg.get("role", "user")).strip()
            content = str(msg.get("content", ""))
            cleaned_messages.append({"role": role, "content": content})

        if hasattr(self.tokenizer, "apply_chat_template"):
            rendered = self.tokenizer.apply_chat_template(
                cleaned_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return rendered

        # fallback
        parts: List[str] = []
        for msg in cleaned_messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"[{role}]\n{content}")
        parts.append("[assistant]\n")
        return "\n\n".join(parts)

    @torch.inference_mode()
    def generate_from_messages(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
        stop: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        assert self.tokenizer is not None
        assert self.model is not None

        prompt_text = self._messages_to_prompt(messages)

        use_max_new_tokens = max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens
        use_temperature = temperature if temperature is not None else self.config.temperature
        use_top_p = top_p if top_p is not None else self.config.top_p
        use_repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        use_do_sample = do_sample if do_sample is not None else self.config.do_sample

        if use_temperature <= 0:
            use_temperature = 0.01
            use_do_sample = False

        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        if cuda_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        start_time = time.time()

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=use_max_new_tokens,
            temperature=use_temperature,
            top_p=use_top_p,
            repetition_penalty=use_repetition_penalty,
            do_sample=use_do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        output_ids = outputs[0]
        gen_ids = output_ids[input_ids.shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        text = trim_stop_text(text, stop).strip()

        elapsed = time.time() - start_time

        return {
            "text": text,
            "usage": {
                "prompt_tokens": int(input_ids.shape[1]),
                "completion_tokens": int(gen_ids.shape[0]),
                "total_tokens": int(input_ids.shape[1] + gen_ids.shape[0]),
            },
            "meta": {
                "elapsed_sec": round(elapsed, 4),
                "model": self.config.model_name,
                "device": self.model_device,
            },
        }


# =========================================================
# openai-like server
# =========================================================


def create_app(engine: LocalQwen3Engine, config: LocalLLMConfig):
    if FastAPI is None:
        raise RuntimeError(
            "FastAPI/uvicorn is not available. Install: pip install fastapi uvicorn"
        )

    app = FastAPI(title="Local LLM Qwen3 API")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "ok": True,
            "model": config.model_name,
            "device": engine.model_device,
            "time": now_unix(),
        }

    @app.get("/v1/models")
    def list_models() -> Dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": config.model_name,
                    "object": "model",
                    "owned_by": "local",
                    "created": now_unix(),
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(payload: Dict[str, Any]) -> Dict[str, Any]:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages is required")

        request_model = str(payload.get("model") or config.model_name).strip()
        if request_model and request_model != config.model_name:
            # 一応受けるけど、ロード済みモデルで返す
            log(f"[LOCAL API] requested model={request_model} but serving loaded model={config.model_name}")

        temperature = safe_float(payload.get("temperature"), config.temperature)
        top_p = safe_float(payload.get("top_p"), config.top_p)
        max_tokens = safe_int(payload.get("max_tokens"), config.max_new_tokens)
        repetition_penalty = safe_float(payload.get("repetition_penalty"), config.repetition_penalty)
        stop = payload.get("stop")
        if isinstance(stop, str):
            stop = [stop]
        elif not isinstance(stop, list):
            stop = None

        do_sample = temperature > 0.0

        result = engine.generate_from_messages(
            messages=messages,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            stop=stop,
        )

        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = now_unix()
        output_text = result["text"]
        usage = result["usage"]

        return {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": config.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }

    @app.post("/generate")
    def generate(payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt = str(payload.get("prompt", ""))
        if not prompt.strip():
            raise HTTPException(status_code=400, detail="prompt is required")

        max_new_tokens = safe_int(payload.get("max_new_tokens"), config.max_new_tokens)
        temperature = safe_float(payload.get("temperature"), config.temperature)
        top_p = safe_float(payload.get("top_p"), config.top_p)

        messages = [{"role": "user", "content": prompt}]
        result = engine.generate_from_messages(
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0.0,
        )
        return result

    @app.exception_handler(Exception)
    async def on_error(_, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(exc),
                    "type": type(exc).__name__,
                }
            },
        )

    return app


# =========================================================
# cli
# =========================================================


def run_cli(engine: LocalQwen3Engine, config: LocalLLMConfig) -> int:
    log("[LOCAL CLI] started")
    log("[LOCAL CLI] exit with Ctrl+C")

    history: List[Dict[str, str]] = []

    try:
        while True:
            user_text = input("user> ").strip()
            if not user_text:
                continue

            history.append({"role": "user", "content": user_text})
            history = clip_history_by_chars(history, config.history_char_budget)

            result = engine.generate_from_messages(
                messages=history,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                do_sample=config.do_sample,
            )

            assistant_text = result["text"].strip()
            history.append({"role": "assistant", "content": assistant_text})
            history = clip_history_by_chars(history, config.history_char_budget)

            print(f"assistant> {assistant_text}")
            log(
                f"[LOCAL CLI] usage prompt={result['usage']['prompt_tokens']} "
                f"completion={result['usage']['completion_tokens']} "
                f"elapsed={result['meta']['elapsed_sec']}s"
            )

    except KeyboardInterrupt:
        log("[LOCAL CLI] stopped")
        return 0


def clip_history_by_chars(messages: List[Dict[str, str]], budget: int) -> List[Dict[str, str]]:
    if budget <= 0:
        return messages

    total = 0
    kept_reversed: List[Dict[str, str]] = []

    for msg in reversed(messages):
        content = str(msg.get("content", ""))
        size = len(content)
        if kept_reversed and total + size > budget:
            break
        kept_reversed.append(msg)
        total += size

    kept_reversed.reverse()
    return kept_reversed


# =========================================================
# args / entry
# =========================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Qwen3 LLM runner (4bit)")

    parser.add_argument(
        "--mode",
        choices=["api", "cli"],
        default="api",
        help="Run as API server or interactive CLI.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen3-4B-Instruct"),
        help="HF model id.",
    )
    parser.add_argument("--host", type=str, default=os.getenv("LOCAL_LLM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("LOCAL_LLM_PORT", "8000")))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--history-char-budget", type=int, default=6000)

    parser.add_argument("--no-4bit", action="store_true", help="Disable 4bit quantization.")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-double-quant", action="store_true")
    parser.add_argument("--quant-type", type=str, default="nf4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = LocalLLMConfig(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        do_sample=args.temperature > 0.0,
        host=args.host,
        port=args.port,
        trust_remote_code=bool(args.trust_remote_code),
        load_in_4bit=not args.no_4bit,
        use_double_quant=not args.no_double_quant,
        quant_type=args.quant_type,
        history_char_budget=args.history_char_budget,
    )

    engine = LocalQwen3Engine(config)
    engine.load()

    if args.mode == "cli":
        raise SystemExit(run_cli(engine, config))

    if uvicorn is None:
        raise RuntimeError("uvicorn is not available. Install: pip install uvicorn fastapi")

    app = create_app(engine, config)

    log(f"[LOCAL API] serving on http://{config.host}:{config.port}")
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()