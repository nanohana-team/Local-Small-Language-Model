from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LearningConfig:
    name: str
    type: str
    model: str
    system_prompt: str
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 192
    device: str = "cuda"
    bf16: bool = True
    fp16: bool = False

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LearningConfig":
        required = ["name", "type", "model"]
        for key in required:
            if key not in d:
                raise ValueError(f"Learning config missing required key: {key}")
        return LearningConfig(
            name=str(d["name"]),
            type=str(d["type"]),
            model=str(d["model"]),
            system_prompt=str(
                d.get(
                    "system_prompt",
                    "あなたは日本語で自然に会話する小型AIです。簡潔で自然な返答をしてください。",
                )
            ),
            temperature=float(d.get("temperature", 0.7)),
            top_p=float(d.get("top_p", 0.9)),
            max_tokens=int(d.get("max_tokens", 192)),
            device=str(d.get("device", "cuda")),
            bf16=bool(d.get("bf16", True)),
            fp16=bool(d.get("fp16", False)),
        )


class LocalStudentClient:
    def __init__(self, cfg: LearningConfig):
        self.cfg = cfg
        self.device = "cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_path = Path(cfg.model)
        if model_path.exists() and (model_path / "adapter_config.json").exists():
            peft_cfg = PeftConfig.from_pretrained(cfg.model)
            dtype = self._resolve_dtype()
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_cfg.base_model_name_or_path,
                torch_dtype=dtype,
            )
            self.model = PeftModel.from_pretrained(base_model, cfg.model)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model,
                torch_dtype=self._resolve_dtype(),
            )

        if self.device == "cuda":
            self.model = self.model.to("cuda")
        self.model.eval()

    def _resolve_dtype(self):
        if self.cfg.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if self.cfg.fp16 and torch.cuda.is_available():
            return torch.float16
        return torch.float32

    def generate(self, messages: List[Dict[str, str]]) -> str:
        prompt_messages = [{"role": "system", "content": self.cfg.system_prompt}, *messages]
        prompt = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_tokens,
                do_sample=True,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        new_token_ids = output_ids[0][prompt_len:]
        text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        return text.split("\n\n")[0].strip() if text else ""
