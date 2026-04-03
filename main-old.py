import os
import re
import json
import time
import uuid
import random
import shutil
import signal
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, Future

from dotenv import load_dotenv

import logging
from logging.handlers import RotatingFileHandler

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
)

load_dotenv()

# =========================
# Optional API clients
# =========================
_HAS_GOOGLE_GENAI = False
_HAS_OPENAI = False

try:
    from google import genai
    _HAS_GOOGLE_GENAI = True
except Exception:
    pass

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    pass


# =========================
# Logging
# =========================
def setup_logger(log_dir: str = "runtime/logs") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("lslm")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "main.log"),
        maxBytes=20 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    error_handler = RotatingFileHandler(
        os.path.join(log_dir, "error.log"),
        maxBytes=20 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(fmt)

    api_eval_handler = RotatingFileHandler(
        os.path.join(log_dir, "api_eval.log"),
        maxBytes=20 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
    )
    api_eval_handler.setLevel(logging.INFO)
    api_eval_handler.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    logger.addHandler(api_eval_handler)
    logger.propagate = False
    return logger


LOGGER = setup_logger()

LIVE_VIEW_PATH = Path("runtime/live_view.txt")
VIEWER_PATH = Path("viewer.py")


def format_model_label(model_id: str) -> str:
    if not model_id:
        return "unknown"
    return model_id.split("/")[-1]


def sanitize_live_view_text(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def append_live_view_block(block: str) -> None:
    ensure_parent_dir(str(LIVE_VIEW_PATH))
    existed = LIVE_VIEW_PATH.exists()
    with open(LIVE_VIEW_PATH, "a", encoding="utf-8") as f:
        if existed and LIVE_VIEW_PATH.stat().st_size > 0:
            f.write("\n\n")
        f.write(block.rstrip() + "\n")


def ensure_viewer_script() -> None:
    viewer_code = """import time
from pathlib import Path
import sys

log_path = Path("runtime/live_view.txt")
last_size = 0

print("live viewer started", flush=True)

while True:
    try:
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            if len(text) > last_size:
                sys.stdout.write(text[last_size:])
                sys.stdout.flush()
                last_size = len(text)
        time.sleep(0.2)
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"[viewer error] {e}", flush=True)
        time.sleep(1.0)
"""
    VIEWER_PATH.write_text(viewer_code, encoding="utf-8")


# ========================
# DPO Datasets
# ========================
class DPODataset(Dataset):
    def __init__(self, path: str, tokenizer, cutoff_len: int):
        self.rows = []
        self.tokenizer = tokenizer
        self.cutoff_len = cutoff_len

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def _encode_pair(self, prompt: str, answer: str):
        prompt_ids = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.cutoff_len,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]

        full = prompt + answer
        tok = self.tokenizer(
            full,
            truncation=True,
            max_length=self.cutoff_len,
            padding=False,
            add_special_tokens=False,
        )

        input_ids = tok["input_ids"]
        attention_mask = tok["attention_mask"]
        labels = input_ids.copy()

        prompt_len = min(len(prompt_ids), len(labels))
        for i in range(prompt_len):
            labels[i] = -100

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __getitem__(self, idx):
        row = self.rows[idx]
        chosen = self._encode_pair(row["prompt"], row["chosen"])
        rejected = self._encode_pair(row["prompt"], row["rejected"])
        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "chosen_labels": chosen["labels"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
            "rejected_labels": rejected["labels"],
            "sample_weight": torch.tensor(float(row.get("sample_weight", 1.0)), dtype=torch.float32),
        }


class DPODataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        def pad_stack(key, pad_value):
            xs = [f[key] for f in features]
            return torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=pad_value)

        return {
            "chosen_input_ids": pad_stack("chosen_input_ids", self.pad_token_id),
            "chosen_attention_mask": pad_stack("chosen_attention_mask", 0),
            "chosen_labels": pad_stack("chosen_labels", -100),
            "rejected_input_ids": pad_stack("rejected_input_ids", self.pad_token_id),
            "rejected_attention_mask": pad_stack("rejected_attention_mask", 0),
            "rejected_labels": pad_stack("rejected_labels", -100),
            "sample_weight": torch.stack([f["sample_weight"] for f in features]),
        }


class SimpleDPOTrainer(Trainer):
    def __init__(self, *args, ref_model=None, beta=0.1, loss_type="sigmoid", **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.beta = beta
        self.loss_type = loss_type
        if self.ref_model is not None:
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False

    def _sequence_logp(self, model, input_ids, attention_mask, labels):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()

        log_probs = F.log_softmax(logits, dim=-1)
        safe_labels = shifted_labels.masked_fill(shifted_labels == -100, 0)
        token_logps = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)

        mask = (shifted_labels != -100).float()
        seq_logp = (token_logps * mask).sum(dim=1)
        return seq_logp

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        chosen_input_ids = inputs["chosen_input_ids"]
        chosen_attention_mask = inputs["chosen_attention_mask"]
        chosen_labels = inputs["chosen_labels"]

        rejected_input_ids = inputs["rejected_input_ids"]
        rejected_attention_mask = inputs["rejected_attention_mask"]
        rejected_labels = inputs["rejected_labels"]

        sample_weight = inputs.get("sample_weight", None)

        pi_chosen = self._sequence_logp(model, chosen_input_ids, chosen_attention_mask, chosen_labels)
        pi_rejected = self._sequence_logp(model, rejected_input_ids, rejected_attention_mask, rejected_labels)

        with torch.no_grad():
            ref_chosen = self._sequence_logp(self.ref_model, chosen_input_ids, chosen_attention_mask, chosen_labels)
            ref_rejected = self._sequence_logp(self.ref_model, rejected_input_ids, rejected_attention_mask, rejected_labels)

        logits = self.beta * ((pi_chosen - pi_rejected) - (ref_chosen - ref_rejected))
        loss_per_sample = -F.logsigmoid(logits)

        if sample_weight is not None:
            sample_weight = sample_weight.to(loss_per_sample.device)
            loss = (loss_per_sample * sample_weight).mean()
        else:
            loss = loss_per_sample.mean()

        return (loss, {"logits": logits}) if return_outputs else loss


# =========================
# Config
# =========================
@dataclass
class AppConfig:
    # ---- Models ----
    student_model_id: str = "google/gemma-3-270m"
    teacher_model_ids: Tuple[str, str] = (
        "Qwen/Qwen2.5-7B-Instruct",
        "LiquidAI/LFM2.5-1.2B-Instruct",
    )

    # ---- Built-in persona tuned for CPU-friendly short replies ----
    builtin_persona: str = """あなたは日本語で応答する軽量会話AIです。
目的は、CPUでも高速に動く短文対話モデルとして振る舞うことです。

ルール:
- 返答は原則1〜2文、長くても3文まで
- まず結論や答えを先に述べる
- 無駄な前置き、重複、言い換えをしない
- 聞かれたことに直接答える
- 不明な点は短く確認するか、分かる範囲だけ答える
- 抽象論より具体的で短い表現を優先する
- 箇条書きは使わず自然な短文で返す
- 絵文字、顔文字、過剰な記号装飾は使わない
- 大げさな感情表現や演技的な語尾は使わない
- 安全で、簡潔で、実用的な返答を優先する

悪い例:
- 長い前置き
- 同じ意味の繰り返し
- 頼まれていない補足の盛りすぎ
- 説明書のような硬すぎる文章

良い例:
- 短く自然
- 要点が先
- 1回で意味が通る
- 低トークンで十分な情報がある"""

    max_new_tokens_teacher: int = 64
    max_new_tokens_student: int = 48
    temperature_teacher: float = 0.7
    temperature_student: float = 0.6
    top_p: float = 0.92
    repetition_penalty: float = 1.10

    # ---- Conversation loop ----
    session_ttl: int = 20
    persist_history_turns: int = 6
    section_size: int = 20
    sessions_per_eval: int = 5
    use_external_eval: bool = True
    endless: bool = True

    # ---- Training ----
    min_overall_to_train: float = 0.56
    min_dpo_margin: float = 0.03
    max_steps_per_cycle: int = 30
    train_batch_size: int = 1
    grad_accum: int = 8
    lr: float = 2e-4
    dpo_lr: float = 1e-5
    dpo_beta: float = 0.1
    dpo_loss_type: str = "sigmoid"
    dpo_enabled: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    cutoff_len: int = 384
    save_every_cycle: bool = True

    # ---- Runtime ----
    seed: int = 42
    cpu_workers: int = max(1, (os.cpu_count() or 4) // 2)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"

    # ---- Paths ----
    root_dir: str = "runtime"
    sessions_dir: str = "runtime/sessions"
    evals_dir: str = "runtime/evals"
    dataset_dir: str = "runtime/datasets"
    dpo_dataset_dir: str = "runtime/dpo_datasets"
    checkpoints_dir: str = "runtime/checkpoints"
    merged_dir: str = "runtime/merged_student"
    latest_link: str = "runtime/latest_student"
    logs_dir: str = "runtime/logs"
    conversations_dir: str = "runtime/logs/conversations"

    # ---- External evaluators ----
    gemini_model: str = "gemini-2.5-flash"
    openai_model: str = "gpt-5-mini"
    gemini_cooldown_sec: int = 300

    # ---- Eval axes ----
    eval_axes: Tuple[str, ...] = (
        "intent_fit",
        "naturalness",
        "consistency",
        "emotional_appropriateness",
        "conciseness",
        "safety",
        "instruction_following",
        "robustness",
    )

    eval_axis_weights: Dict[str, float] = field(default_factory=lambda: {
        "intent_fit": 1.25,
        "naturalness": 1.30,
        "consistency": 1.00,
        "emotional_appropriateness": 0.85,
        "conciseness": 1.35,
        "safety": 1.30,
        "instruction_following": 1.35,
        "robustness": 1.00,
    })

    sample_weight_floor: float = 0.25
    sample_weight_ceiling: float = 3.00
    margin_bonus_scale: float = 0.80
    student_win_bonus: float = 0.35


# =========================
# CPU-oriented seed prompts
# =========================
SEED_PROMPTS = [
    "短く自然に自己紹介して。",
    "一言で要点だけ教えて。",
    "それを2文で説明すると？",
    "結論から先に言うとどうなる？",
    "短く分かりやすく答えて。",
    "長くしないで要点だけ教えて。",
    "初心者向けに短く説明して。",
    "それって結局どういうこと？",
    "一番大事な点だけ言うと？",
    "迷ってる人向けに短く助言して。",
    "おすすめを一つだけ挙げるなら？",
    "比較するなら違いだけ短く教えて。",
    "それのメリットだけ簡潔に言って。",
    "逆にデメリットだけ短く言って。",
    "今すぐ使える形で答えて。",
    "実用目線で一言アドバイスして。",
    "短文で自然に返して。",
    "説明しすぎずに答えて。",
    "会話っぽく短く返すなら？",
    "余計な前置きなしで答えて。",
    "仕事で疲れた。短く労って。",
    "今日やることが多い。短く整理して。",
    "眠いけど作業したい。ひとこと助言して。",
    "集中したい。すぐできる対策を短く教えて。",
    "朝がつらい。短く声をかけて。",
    "ちょっと不安。落ち着く一言をちょうだい。",
    "お昼ごはん迷ってる。軽く提案して。",
    "今の話をもっと短く言い換えて。",
    "同じ意味で自然に言い換えて。",
    "曖昧な質問にも短く答えてみて。",
    "分からない時の自然な返し方を見せて。",
    "相手を待たせない返答ってどんな感じ？",
    "会話を続けやすい短文ってどんなの？",
    "説明調にならない短文のコツは？",
    "CPUで軽く動く会話AIに必要なことは？",
    "軽量モデルで大事なのは速度と質のどっち？",
    "短文AIの学習で重要なデータって何？",
    "小型モデルの弱点を短くまとめて。",
    "軽量モデルでも自然に見せるコツは？",
    "少ない語数で印象を良くするには？",
]


# =========================
# Utils
# =========================
def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def ensure_dirs(cfg: AppConfig) -> None:
    for p in [
        cfg.root_dir,
        cfg.sessions_dir,
        cfg.evals_dir,
        cfg.dataset_dir,
        cfg.dpo_dataset_dir,
        cfg.checkpoints_dir,
        cfg.logs_dir,
        cfg.conversations_dir,
    ]:
        Path(p).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def short_text(text: str, max_len: int = 120) -> str:
    text = (text or "").replace("\n", "\\n")
    return text[:max_len] + ("..." if len(text) > max_len else "")


def atomic_write_json(path: str, data: Any) -> None:
    ensure_parent_dir(path)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def atomic_write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def atomic_write_text(path: str, text: str) -> None:
    ensure_parent_dir(path)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def cpu_flush_session_log(args: Tuple[str, Dict[str, Any]]) -> str:
    path, data = args
    atomic_write_json(path, data)
    return path


def cpu_flush_jsonl(args: Tuple[str, List[Dict[str, Any]]]) -> str:
    path, rows = args
    atomic_write_jsonl(path, rows)
    return path


def cpu_flush_text(args: Tuple[str, str]) -> str:
    path, text = args
    atomic_write_text(path, text)
    return path


# =========================
# Dataset
# =========================
class WeightedSFTJsonlDataset(Dataset):
    def __init__(self, path: str, tokenizer, cutoff_len: int):
        self.rows: List[Dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.cutoff_len = cutoff_len

        LOGGER.info(f"[DATASET_LOAD_START] path={path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        LOGGER.info(f"[DATASET_LOAD_DONE] path={path} rows={len(self.rows)}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        prompt_text = row["prompt"]
        chosen_text = row["chosen"]

        prompt_ids = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.cutoff_len,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )["input_ids"]

        full_text = prompt_text + chosen_text
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.cutoff_len,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        labels = input_ids.copy()

        prompt_len = min(len(prompt_ids), len(labels))
        for i in range(prompt_len):
            labels[i] = -100

        sample_weight = float(row.get("sample_weight", 1.0))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "sample_weight": torch.tensor(sample_weight, dtype=torch.float32),
        }


# =========================
# Weighted Trainer
# =========================
class WeightedSFTDataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [f["input_ids"] for f in features],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [f["attention_mask"] for f in features],
            batch_first=True,
            padding_value=0,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [f["labels"] for f in features],
            batch_first=True,
            padding_value=-100,
        )
        sample_weight = torch.stack([f["sample_weight"] for f in features])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sample_weight": sample_weight,
        }


class WeightedSFTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        sample_weight = inputs.pop("sample_weight", None)

        outputs = model(**inputs)
        logits = outputs.get("logits")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        vocab_size = shift_logits.size(-1)

        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(shift_labels.size())

        valid_mask = (shift_labels != -100).float()
        token_loss_sum = (loss_per_token * valid_mask).sum(dim=1)
        token_count = valid_mask.sum(dim=1).clamp_min(1.0)
        loss_per_sample = token_loss_sum / token_count

        if sample_weight is not None:
            sample_weight = sample_weight.to(loss_per_sample.device)
            loss = (loss_per_sample * sample_weight).mean()
        else:
            loss = loss_per_sample.mean()

        return (loss, outputs) if return_outputs else loss


# =========================
# Model wrappers
# =========================
class TextModel:
    def __init__(self, model_id: str, dtype: torch.dtype):
        self.model_id = model_id
        self.dtype = dtype

        LOGGER.info(f"[MODEL_LOAD_START] tokenizer model={model_id}")
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN"),
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        LOGGER.info(f"[MODEL_LOAD_DONE] tokenizer model={model_id} sec={time.perf_counter() - t0:.2f}")

        LOGGER.info(f"[MODEL_LOAD_START] weights model={model_id} dtype={dtype}")
        t1 = time.perf_counter()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="cuda" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN"),
        )
        self.model.eval()
        LOGGER.info(
            f"[MODEL_LOAD_DONE] weights model={model_id} sec={time.perf_counter() - t1:.2f} "
            f"device={self.model.device}"
        )

    def _build_prompt(self, system_prompt: str, history: List[Dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(history)
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as e:
                LOGGER.warning(f"[CHAT_TEMPLATE_FAIL] model={self.model_id} error={e}")

        lines = [f"[SYSTEM]\n{system_prompt}\n"]
        for m in history:
            lines.append(f"[{m['role'].upper()}]\n{m['content']}\n")
        lines.append("[ASSISTANT]\n")
        return "\n".join(lines)

    @torch.no_grad()
    def generate(
        self,
        system_prompt: str,
        history: List[Dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> str:
        prompt = self._build_prompt(system_prompt, history)
        t0 = time.perf_counter()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)
        input_tokens = int(inputs["input_ids"].shape[1])

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if not generation_kwargs["do_sample"]:
            generation_kwargs.pop("temperature", None)
            generation_kwargs.pop("top_p", None)

        out = self.model.generate(**inputs, **generation_kwargs)
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        sec = time.perf_counter() - t0

        LOGGER.info(
            f"[GEN_DONE] model={self.model_id} in_tokens={input_tokens} out_tokens={len(gen_ids)} "
            f"out_chars={len(text)} sec={sec:.2f} text={short_text(text, 180)}"
        )
        return text


class StudentModel:
    def __init__(self, model_id: str, dtype: torch.dtype):
        self.base_model_id = model_id
        self.current_model_path = model_id
        self.dtype = dtype
        self.model = None

        LOGGER.info(f"[STUDENT_LOAD_START] tokenizer model={model_id}")
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN"),
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        LOGGER.info(f"[STUDENT_LOAD_DONE] tokenizer model={model_id} sec={time.perf_counter() - t0:.2f}")

        self._load_model_from_path(model_id)

    def _load_model_from_path(self, model_path: str):
        LOGGER.info(f"[STUDENT_WEIGHTS_LOAD_START] model={model_path} dtype={self.dtype}")
        t0 = time.perf_counter()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            device_map="cuda" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN"),
        )
        self.model.eval()
        LOGGER.info(
            f"[STUDENT_WEIGHTS_LOAD_DONE] model={model_path} sec={time.perf_counter() - t0:.2f} "
            f"device={self.model.device}"
        )

    def ensure_loaded(self):
        if self.model is None:
            LOGGER.warning(f"[STUDENT_AUTO_RECOVER] reloading current_model_path={self.current_model_path}")
            self._load_model_from_path(self.current_model_path)

    def unload(self):
        LOGGER.info(f"[STUDENT_UNLOAD] current_model_path={self.current_model_path}")
        try:
            if self.model is not None:
                del self.model
        except Exception:
            pass
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reload_from(self, model_path: str):
        LOGGER.info(f"[STUDENT_RELOAD_START] model_path={model_path}")
        self.unload()
        self.current_model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN"),
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._load_model_from_path(model_path)
        LOGGER.info(f"[STUDENT_RELOAD_DONE] current_model_path={self.current_model_path}")

    def _build_prompt(self, system_prompt: str, history: List[Dict[str, str]]) -> str:
        chat_template = getattr(self.tokenizer, "chat_template", None)

        if chat_template:
            try:
                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(history)
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as e:
                LOGGER.warning(f"[STUDENT_CHAT_TEMPLATE_FAIL] model={self.current_model_path} error={e}")

        lines = [f"[SYSTEM]\n{system_prompt}\n"]
        for m in history:
            lines.append(f"[{m['role'].upper()}]\n{m['content']}\n")
        lines.append("[ASSISTANT]\n")
        return "\n".join(lines)

    @torch.no_grad()
    def generate(
        self,
        system_prompt: str,
        history: List[Dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> str:
        self.ensure_loaded()
        prompt = self._build_prompt(system_prompt, history)
        t0 = time.perf_counter()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)
        input_tokens = int(inputs["input_ids"].shape[1])

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if not generation_kwargs["do_sample"]:
            generation_kwargs.pop("temperature", None)
            generation_kwargs.pop("top_p", None)

        out = self.model.generate(**inputs, **generation_kwargs)
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        sec = time.perf_counter() - t0

        LOGGER.info(
            f"[STUDENT_GEN_DONE] model={self.current_model_path} in_tokens={input_tokens} "
            f"out_tokens={len(gen_ids)} out_chars={len(text)} sec={sec:.2f} text={short_text(text, 180)}"
        )
        return text

    def _make_peft_config(self, lora_r: int, lora_alpha: int, lora_dropout: float) -> LoraConfig:
        return LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )

    def _merge_adapter_dir(self, previous_model_path: str, adapter_dir: str, merged_dir: str) -> str:
        LOGGER.info(f"[MERGE_START] base={previous_model_path} adapter_dir={adapter_dir} merged_dir={merged_dir}")
        base_for_merge = AutoModelForCausalLM.from_pretrained(
            previous_model_path,
            torch_dtype=self.dtype,
            device_map="cuda" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN"),
        )
        merged = PeftModel.from_pretrained(base_for_merge, adapter_dir)
        merged = merged.merge_and_unload()
        merged.save_pretrained(merged_dir)
        self.tokenizer.save_pretrained(merged_dir)
        LOGGER.info(f"[MERGE_DONE] merged_dir={merged_dir}")
        return merged_dir

    def train_lora_and_merge_sft(
        self,
        dataset_path: str,
        out_dir: str,
        cutoff_len: int,
        batch_size: int,
        grad_accum: int,
        learning_rate: float,
        max_steps: int,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
    ) -> str:
        LOGGER.info(
            f"[SFT_START] model={self.current_model_path} dataset={dataset_path} out_dir={out_dir} "
            f"cutoff_len={cutoff_len} batch_size={batch_size} grad_accum={grad_accum} "
            f"lr={learning_rate} max_steps={max_steps}"
        )

        previous_model_path = self.current_model_path
        self.unload()

        try:
            Path(out_dir).mkdir(parents=True, exist_ok=True)

            train_model = AutoModelForCausalLM.from_pretrained(
                previous_model_path,
                torch_dtype=self.dtype,
                device_map="cuda" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                token=os.getenv("HF_TOKEN"),
            )
            train_model = get_peft_model(train_model, self._make_peft_config(lora_r, lora_alpha, lora_dropout))

            train_ds = WeightedSFTJsonlDataset(dataset_path, self.tokenizer, cutoff_len)
            LOGGER.info(f"[SFT_DATASET] rows={len(train_ds)} dataset={dataset_path}")

            args = TrainingArguments(
                output_dir=out_dir,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,
                learning_rate=learning_rate,
                max_steps=max_steps,
                logging_steps=1,
                save_steps=max_steps,
                save_total_limit=1,
                bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
                report_to=[],
                remove_unused_columns=False,
            )

            trainer = WeightedSFTTrainer(
                model=train_model,
                args=args,
                train_dataset=train_ds,
                data_collator=WeightedSFTDataCollator(self.tokenizer.pad_token_id),
            )
            trainer.train()
            trainer.save_model(out_dir)

            merged_dir = os.path.join(out_dir, "merged")
            merged_dir = self._merge_adapter_dir(previous_model_path, out_dir, merged_dir)
            self.reload_from(merged_dir)
            LOGGER.info(f"[SFT_END] current_model_path={self.current_model_path}")
            return merged_dir

        except Exception as e:
            LOGGER.error(f"[SFT_FAIL] error={e}", exc_info=True)
            self.current_model_path = previous_model_path
            self._load_model_from_path(previous_model_path)
            raise

    def train_dpo_and_merge(
        self,
        dataset_path: str,
        out_dir: str,
        cutoff_len: int,
        batch_size: int,
        grad_accum: int,
        learning_rate: float,
        max_steps: int,
        beta: float,
        loss_type: str,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
    ) -> str:
        LOGGER.info(
            f"[DPO_START] model={self.current_model_path} dataset={dataset_path} out_dir={out_dir} "
            f"cutoff_len={cutoff_len} batch_size={batch_size} grad_accum={grad_accum} "
            f"lr={learning_rate} max_steps={max_steps} beta={beta} loss_type={loss_type}"
        )

        previous_model_path = self.current_model_path
        self.unload()

        try:
            Path(out_dir).mkdir(parents=True, exist_ok=True)

            policy_model = AutoModelForCausalLM.from_pretrained(
                previous_model_path,
                torch_dtype=self.dtype,
                device_map="cuda" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                token=os.getenv("HF_TOKEN"),
            )
            ref_model = AutoModelForCausalLM.from_pretrained(
                previous_model_path,
                torch_dtype=self.dtype,
                device_map="cuda" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                token=os.getenv("HF_TOKEN"),
            )

            policy_model = get_peft_model(
                policy_model,
                self._make_peft_config(lora_r, lora_alpha, lora_dropout),
            )

            train_ds = DPODataset(dataset_path, self.tokenizer, cutoff_len)
            LOGGER.info(f"[DPO_DATASET] rows={len(train_ds)} dataset={dataset_path}")

            args = TrainingArguments(
                output_dir=out_dir,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,
                learning_rate=learning_rate,
                max_steps=max_steps,
                logging_steps=1,
                save_steps=max_steps,
                save_total_limit=1,
                bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
                report_to=[],
                remove_unused_columns=False,
            )

            trainer = SimpleDPOTrainer(
                model=policy_model,
                ref_model=ref_model,
                beta=beta,
                loss_type=loss_type,
                args=args,
                train_dataset=train_ds,
                data_collator=DPODataCollator(self.tokenizer.pad_token_id),
            )

            trainer.train()
            trainer.save_model(out_dir)

            merged_dir = os.path.join(out_dir, "merged")
            merged_dir = self._merge_adapter_dir(previous_model_path, out_dir, merged_dir)

            self.reload_from(merged_dir)
            LOGGER.info(f"[DPO_END] current_model_path={self.current_model_path}")
            return merged_dir

        except Exception as e:
            LOGGER.error(f"[DPO_FAIL] error={e}", exc_info=True)
            self.current_model_path = previous_model_path
            self._load_model_from_path(previous_model_path)
            raise


# =========================
# Evaluator
# =========================
class ExternalEvaluator:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.gemini_client = None
        self.openai_client = None
        self.gemini_disabled_until = 0.0

        if _HAS_GOOGLE_GENAI and os.getenv("GEMINI_API_KEY"):
            try:
                self.gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
                LOGGER.info("[EVAL_INIT] Gemini client enabled")
            except Exception as e:
                LOGGER.warning(f"[EVAL_INIT_FAIL] Gemini client unavailable: {e}")

        if _HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
            try:
                self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                LOGGER.info("[EVAL_INIT] OpenAI client enabled")
            except Exception as e:
                LOGGER.warning(f"[EVAL_INIT_FAIL] OpenAI client unavailable: {e}")

    def _weighted_mean(self, axis_scores: Dict[str, float]) -> float:
        total_w = 0.0
        total = 0.0
        for axis in self.cfg.eval_axes:
            w = float(self.cfg.eval_axis_weights.get(axis, 1.0))
            v = float(axis_scores.get(axis, 0.0))
            total_w += w
            total += v * w
        return 0.0 if total_w <= 0 else total / total_w

    def _extract_retry_delay_sec(self, message: str) -> int:
        patterns = [
            r"retry in\s+([0-9]+(?:\.[0-9]+)?)s",
            r"'retryDelay':\s*'([0-9]+)s'",
            r'"retryDelay":\s*"([0-9]+)s"',
        ]
        for pattern in patterns:
            m = re.search(pattern, message, flags=re.IGNORECASE)
            if m:
                try:
                    return max(1, int(float(m.group(1))))
                except Exception:
                    pass
        return 0

    def _is_quota_error(self, message: str) -> bool:
        lowered = message.lower()
        return (
            "429" in message
            or "resource_exhausted" in lowered
            or "quota exceeded" in lowered
            or "rate limit" in lowered
        )

    def _log_eval_request_preview(self, provider: str, items: List[Dict[str, str]]) -> None:
        try:
            preview = {
                "provider": provider,
                "count": len(items),
                "first_prompt": short_text(items[0].get("prompt", ""), 240) if items else "",
                "first_teacher_a": short_text(items[0].get("teacher_a", ""), 240) if items else "",
                "first_teacher_b": short_text(items[0].get("teacher_b", ""), 240) if items else "",
                "first_student": short_text(items[0].get("student", ""), 240) if items else "",
            }
            LOGGER.info(f"[API_EVAL_REQUEST] {json.dumps(preview, ensure_ascii=False)}")
        except Exception as e:
            LOGGER.warning(f"[API_EVAL_REQUEST_LOG_FAIL][{provider}] {e}")

    def _log_eval_response_summary(self, provider: str, results: List[Dict[str, Any]], raw_text: Optional[str] = None) -> None:
        try:
            winners = {}
            for r in results:
                winners[r["winner"]] = winners.get(r["winner"], 0) + 1
            payload = {
                "provider": provider,
                "count": len(results),
                "winner_counts": winners,
                "first_overall_scores": results[0].get("overall_scores", {}) if results else {},
                "first_reason": short_text(str(results[0].get("reason", "")), 240) if results else "",
            }
            LOGGER.info(f"[API_EVAL_RESPONSE] {json.dumps(payload, ensure_ascii=False)}")
            if raw_text is not None:
                LOGGER.info(f"[API_EVAL_RAW][{provider}] {short_text(raw_text, 4000)}")
        except Exception as e:
            LOGGER.warning(f"[API_EVAL_RESPONSE_LOG_FAIL][{provider}] {e}")

    def evaluate_section(self, items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        if not items:
            return []

        LOGGER.info(f"[EVAL_SECTION_START] count={len(items)}")
        now = time.time()

        if self.gemini_client is not None and now < self.gemini_disabled_until:
            remaining = max(0, int(self.gemini_disabled_until - now))
            LOGGER.info(f"[EVAL_GEMINI_SKIPPED] cooldown_remaining_sec={remaining}")
        elif self.cfg.use_external_eval and self.gemini_client is not None:
            try:
                self._log_eval_request_preview("Gemini", items)
                return self._eval_section_with_gemini(items)
            except Exception as e:
                msg = str(e)
                if self._is_quota_error(msg):
                    retry_sec = self._extract_retry_delay_sec(msg)
                    cooldown_sec = retry_sec + 5 if retry_sec > 0 else min(self.cfg.gemini_cooldown_sec, 300)
                    self.gemini_disabled_until = time.time() + cooldown_sec
                    LOGGER.warning(f"[EVAL_SECTION_FAIL][GeminiQuota] cooldown_sec={cooldown_sec} message={msg}")
                else:
                    LOGGER.warning(f"[EVAL_SECTION_FAIL][Gemini] {msg}", exc_info=True)

        if self.cfg.use_external_eval and self.openai_client is not None:
            try:
                self._log_eval_request_preview("OpenAI", items)
                return self._eval_section_with_openai(items)
            except Exception as e:
                LOGGER.warning(f"[EVAL_SECTION_FAIL][OpenAI] {e}", exc_info=True)

        LOGGER.info("[EVAL_SECTION_FALLBACK] using local heuristic evaluator")
        results = [self._local_fallback_eval(x) for x in items]
        self._log_eval_response_summary("LocalFallback", results)
        return results

    def _build_section_eval_prompt(self, items: List[Dict[str, str]]) -> str:
        axes_text = "\n".join([
            "- intent_fit: ユーザー意図への適合性",
            "- naturalness: 日本語としての自然さ",
            "- consistency: 文脈と内容の一貫性",
            "- emotional_appropriateness: 感情トーンや距離感の適切さ",
            "- conciseness: 不要に長すぎず簡潔か",
            "- safety: 安全で問題ないか",
            "- instruction_following: 指示や条件の遵守",
            "- robustness: 曖昧入力に対する頑健さ",
        ])

        blocks = []
        for i, x in enumerate(items):
            blocks.append(f"""
TURN_INDEX: {i}
PROMPT:
{x['prompt']}

CANDIDATE_TEACHER_A:
{x['teacher_a']}

CANDIDATE_TEACHER_B:
{x['teacher_b']}

CANDIDATE_STUDENT:
{x['student']}
""".strip())

        joined = "\n\n====================\n\n".join(blocks)

        return f"""
あなたは会話品質評価器です。
複数ターン分の候補をまとめて評価してください。
各ターンごとに、3候補 teacher_a / teacher_b / student を比較し、
8軸を 0.0000〜1.0000 で採点してください。

重視する点:
- 意図に直接答えているか
- 日本語として自然か
- 短く無駄が少ないか
- 指示どおり簡潔に返しているか
- 小型実用会話AIとして扱いやすいか

必ず JSON のみを返してください。説明文やコードフェンスは禁止です。

評価軸:
{axes_text}

返却形式:
{{
  "results": [
    {{
      "turn_index": 0,
      "winner": "teacher_a" | "teacher_b" | "student",
      "axis_scores": {{
        "teacher_a": {{
          "intent_fit": 0.0000,
          "naturalness": 0.0000,
          "consistency": 0.0000,
          "emotional_appropriateness": 0.0000,
          "conciseness": 0.0000,
          "safety": 0.0000,
          "instruction_following": 0.0000,
          "robustness": 0.0000
        }},
        "teacher_b": {{
          "intent_fit": 0.0000,
          "naturalness": 0.0000,
          "consistency": 0.0000,
          "emotional_appropriateness": 0.0000,
          "conciseness": 0.0000,
          "safety": 0.0000,
          "instruction_following": 0.0000,
          "robustness": 0.0000
        }},
        "student": {{
          "intent_fit": 0.0000,
          "naturalness": 0.0000,
          "consistency": 0.0000,
          "emotional_appropriateness": 0.0000,
          "conciseness": 0.0000,
          "safety": 0.0000,
          "instruction_following": 0.0000,
          "robustness": 0.0000
        }}
      }},
      "reason": "1〜2文"
    }}
  ]
}}

評価対象:
{joined}
""".strip()

    def _eval_section_with_gemini(self, items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        prompt = self._build_section_eval_prompt(items)
        resp = self.gemini_client.models.generate_content(
            model=self.cfg.gemini_model,
            contents=prompt,
        )
        text = getattr(resp, "text", None)
        if not text:
            raise RuntimeError("Gemini returned empty text")
        results = self._safe_parse_section_eval_json(text, items)
        self._log_eval_response_summary("Gemini", results, raw_text=text)
        return results

    def _eval_section_with_openai(self, items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        prompt = self._build_section_eval_prompt(items)
        resp = self.openai_client.responses.create(
            model=self.cfg.openai_model,
            input=prompt,
        )
        text = getattr(resp, "output_text", None)
        if not text:
            raise RuntimeError("OpenAI returned empty text")
        results = self._safe_parse_section_eval_json(text, items)
        self._log_eval_response_summary("OpenAI", results, raw_text=text)
        return results

    def _local_fallback_eval(self, payload: Dict[str, str]) -> Dict[str, Any]:
        def score_text_axes(x: str) -> Dict[str, float]:
            text = (x or "").strip()
            length = len(text)
            words = text.split()
            uniq_ratio = len(set(words)) / max(1, len(words))

            naturalness = 0.58 + min(0.22, uniq_ratio * 0.22)
            intent_fit = 0.62 if length >= 8 else 0.40
            consistency = 0.58 if length >= 10 else 0.42
            emotional_appropriateness = 0.62
            conciseness = 0.92 if 8 <= length <= 90 else 0.55
            safety = 0.95
            instruction_following = 0.70 if 8 <= length <= 100 else 0.35
            robustness = 0.55

            if length > 140:
                naturalness -= 0.08
                conciseness -= 0.30
                instruction_following -= 0.15

            return {
                "intent_fit": float(max(0.0, min(1.0, intent_fit))),
                "naturalness": float(max(0.0, min(1.0, naturalness))),
                "consistency": float(max(0.0, min(1.0, consistency))),
                "emotional_appropriateness": float(max(0.0, min(1.0, emotional_appropriateness))),
                "conciseness": float(max(0.0, min(1.0, conciseness))),
                "safety": float(max(0.0, min(1.0, safety))),
                "instruction_following": float(max(0.0, min(1.0, instruction_following))),
                "robustness": float(max(0.0, min(1.0, robustness))),
            }

        axis_scores = {
            "teacher_a": score_text_axes(payload["teacher_a"]),
            "teacher_b": score_text_axes(payload["teacher_b"]),
            "student": score_text_axes(payload["student"]),
        }
        overall_scores = {k: self._weighted_mean(v) for k, v in axis_scores.items()}
        winner = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)[0][0]
        return {
            "winner": winner,
            "axis_scores": axis_scores,
            "overall_scores": overall_scores,
            "reason": "Fallback heuristic evaluation used because no external evaluator was available.",
        }

    def _safe_parse_section_eval_json(self, text: str, items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json", "", 1).strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start:end + 1]

        data = json.loads(cleaned)
        results = data.get("results", [])
        normalized: List[Optional[Dict[str, Any]]] = [None] * len(items)

        for item in results:
            try:
                idx = int(item.get("turn_index"))
            except Exception:
                continue
            if not (0 <= idx < len(items)):
                continue

            axis_scores_raw = item.get("axis_scores", {})
            axis_scores: Dict[str, Dict[str, float]] = {}
            for model_name in ("teacher_a", "teacher_b", "student"):
                one = axis_scores_raw.get(model_name, {})
                axis_scores[model_name] = {}
                for axis in self.cfg.eval_axes:
                    try:
                        val = float(one.get(axis, 0.0))
                    except Exception:
                        val = 0.0
                    axis_scores[model_name][axis] = max(0.0, min(1.0, val))

            overall_scores = {
                name: self._weighted_mean(axis_scores[name])
                for name in ("teacher_a", "teacher_b", "student")
            }
            winner = item.get("winner")
            if winner not in ("teacher_a", "teacher_b", "student"):
                winner = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)[0][0]

            normalized[idx] = {
                "winner": winner,
                "axis_scores": axis_scores,
                "overall_scores": overall_scores,
                "reason": str(item.get("reason", "")),
            }

        out: List[Dict[str, Any]] = []
        for i in range(len(items)):
            if normalized[i] is None:
                out.append(self._local_fallback_eval(items[i]))
            else:
                out.append(normalized[i])
        return out


# =========================
# Main runtime
# =========================
class EndlessTrainerRuntime:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        ensure_dirs(cfg)
        set_seed(cfg.seed)

        self.persona_text = cfg.builtin_persona

        LOGGER.info(f"[BOOT] cwd={os.getcwd()}")
        LOGGER.info(f"[BOOT] config={json.dumps(asdict(cfg), ensure_ascii=False)}")
        LOGGER.info(f"[BOOT] pythonpid={os.getpid()} cuda_available={torch.cuda.is_available()} device={cfg.device}")

        if cfg.device != "cuda":
            LOGGER.warning("CUDA is not available. Training will be very slow on CPU.")

        dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float16

        LOGGER.info(f"[BOOT] Loading teachers on GPU/CPU: {cfg.teacher_model_ids}")
        self.teacher_a = TextModel(cfg.teacher_model_ids[0], dtype=dtype)
        self.teacher_b = TextModel(cfg.teacher_model_ids[1], dtype=dtype)

        LOGGER.info(f"[BOOT] Loading student on GPU/CPU: {cfg.student_model_id}")
        self.student = StudentModel(cfg.student_model_id, dtype=dtype)

        LOGGER.info("[BOOT] Initializing external evaluator")
        self.evaluator = ExternalEvaluator(cfg)

        LOGGER.info(f"[BOOT] Starting CPU worker pool: workers={cfg.cpu_workers}")
        self.cpu_pool = ProcessPoolExecutor(max_workers=cfg.cpu_workers)
        self.cycle_index = 0
        self.keep_running = True
        self.teacher_a_label = format_model_label(cfg.teacher_model_ids[0])
        self.teacher_b_label = format_model_label(cfg.teacher_model_ids[1])
        self.student_label = format_model_label(cfg.student_model_id)
        self.pending_sessions: List[Dict[str, Any]] = []

        LIVE_VIEW_PATH.write_text("", encoding="utf-8")

    def _submit_with_logging(self, fn, args, label: str) -> Future:
        future = self.cpu_pool.submit(fn, args)

        def _done_callback(f: Future):
            try:
                result = f.result()
                LOGGER.info(f"[ASYNC_SAVE_DONE] label={label} path={result}")
            except Exception as e:
                LOGGER.error(f"[ASYNC_SAVE_FAIL] label={label} error={e}", exc_info=True)

        future.add_done_callback(_done_callback)
        return future

    def shutdown(self):
        self.keep_running = False
        LOGGER.info("[SHUTDOWN] shutting down CPU pool")
        try:
            self.cpu_pool.shutdown(wait=True, cancel_futures=False)
        except Exception as e:
            LOGGER.error(f"[SHUTDOWN_FAIL] {e}", exc_info=True)

    def run(self):
        def _sig_handler(signum, frame):
            LOGGER.info(f"[SIGNAL] received signal={signum}, stopping...")
            self.keep_running = False

        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)

        while self.keep_running:
            self.cycle_index += 1
            cycle_id = f"cycle_{self.cycle_index:06d}_{uuid.uuid4().hex[:8]}"
            LOGGER.info(f"[CYCLE_START] cycle_id={cycle_id} ts={now_ts()}")

            try:
                self.student.ensure_loaded()
                session = self._run_one_session(cycle_id)
                self.pending_sessions.append(session)

                LOGGER.info(
                    f"[SESSION_BUFFER] buffered={len(self.pending_sessions)}/{self.cfg.sessions_per_eval} "
                    f"latest_cycle_id={cycle_id}"
                )

                if len(self.pending_sessions) >= self.cfg.sessions_per_eval:
                    batch_id = f"batch_{self.cycle_index:06d}_{uuid.uuid4().hex[:8]}"
                    LOGGER.info(
                        f"[BATCH_EVAL_START] batch_id={batch_id} session_count={len(self.pending_sessions)} "
                        f"cycles={[s['cycle_id'] for s in self.pending_sessions]}"
                    )

                    sft_path, dpo_path, best_source = self._build_datasets_from_sessions(batch_id, self.pending_sessions)

                    if sft_path is not None:
                        sft_ckpt_dir = os.path.join(self.cfg.checkpoints_dir, batch_id, "sft")
                        LOGGER.info(f"[BATCH_SFT] batch_id={batch_id} dataset_path={sft_path} best_source={best_source}")
                        merged_after_sft = self.student.train_lora_and_merge_sft(
                            dataset_path=sft_path,
                            out_dir=sft_ckpt_dir,
                            cutoff_len=self.cfg.cutoff_len,
                            batch_size=self.cfg.train_batch_size,
                            grad_accum=self.cfg.grad_accum,
                            learning_rate=self.cfg.lr,
                            max_steps=min(self.cfg.max_steps_per_cycle, max(1, self._count_jsonl_rows(sft_path) * 2)),
                            lora_r=self.cfg.lora_r,
                            lora_alpha=self.cfg.lora_alpha,
                            lora_dropout=self.cfg.lora_dropout,
                        )
                        self._refresh_latest_symlink_or_copy(merged_after_sft)

                    if self.cfg.dpo_enabled and dpo_path is not None:
                        dpo_ckpt_dir = os.path.join(self.cfg.checkpoints_dir, batch_id, "dpo")
                        LOGGER.info(f"[BATCH_DPO] batch_id={batch_id} dataset_path={dpo_path}")
                        merged_dir = self.student.train_dpo_and_merge(
                            dataset_path=dpo_path,
                            out_dir=dpo_ckpt_dir,
                            cutoff_len=self.cfg.cutoff_len,
                            batch_size=self.cfg.train_batch_size,
                            grad_accum=self.cfg.grad_accum,
                            learning_rate=self.cfg.dpo_lr,
                            max_steps=min(self.cfg.max_steps_per_cycle, max(1, self._count_jsonl_rows(dpo_path) * 2)),
                            beta=self.cfg.dpo_beta,
                            loss_type=self.cfg.dpo_loss_type,
                            lora_r=self.cfg.lora_r,
                            lora_alpha=self.cfg.lora_alpha,
                            lora_dropout=self.cfg.lora_dropout,
                        )
                        self._refresh_latest_symlink_or_copy(merged_dir)
                    elif sft_path is None:
                        LOGGER.info(f"[BATCH_SKIP_TRAIN] batch_id={batch_id} reason=no_trainable_dataset")

                    self.pending_sessions = []
                    LOGGER.info(f"[BATCH_EVAL_DONE] batch_id={batch_id}")

            except Exception as e:
                LOGGER.error(f"[CYCLE_FAIL] cycle_id={cycle_id} error={e}", exc_info=True)
                try:
                    self.student.ensure_loaded()
                except Exception as recover_e:
                    LOGGER.error(f"[CYCLE_RECOVER_FAIL] cycle_id={cycle_id} error={recover_e}", exc_info=True)

            LOGGER.info(f"[CYCLE_END] cycle_id={cycle_id} ts={now_ts()}")
            time.sleep(1.0)
            if not self.cfg.endless:
                break

        if self.pending_sessions:
            LOGGER.info(
                f"[SHUTDOWN_BUFFER_REMAINING] remaining_sessions={len(self.pending_sessions)} "
                f"cycles={[s['cycle_id'] for s in self.pending_sessions]}"
            )

    def _count_jsonl_rows(self, path: str) -> int:
        n = 0
        with open(path, "r", encoding="utf-8") as f:
            for _ in f:
                n += 1
        LOGGER.info(f"[COUNT_JSONL_DONE] path={path} rows={n}")
        return n

    def _refresh_latest_symlink_or_copy(self, merged_dir: str):
        latest = Path(self.cfg.latest_link)
        LOGGER.info(f"[LATEST_REFRESH_START] merged_dir={merged_dir} latest={latest}")
        if latest.exists() or latest.is_symlink():
            if latest.is_symlink():
                latest.unlink()
            else:
                shutil.rmtree(latest)
        try:
            latest.symlink_to(Path(merged_dir).resolve(), target_is_directory=True)
            LOGGER.info(f"[LATEST_REFRESH_DONE] method=symlink latest={latest}")
        except Exception as e:
            LOGGER.warning(f"[LATEST_SYMLINK_FAIL] {e}; fallback=copytree")
            shutil.copytree(merged_dir, latest)
            LOGGER.info(f"[LATEST_REFRESH_DONE] method=copytree latest={latest}")

    def _render_conversation_text(self, session: Dict[str, Any]) -> str:
        lines = [
            f"cycle_id: {session['cycle_id']}",
            f"created_at: {session['created_at']}",
            f"seed_prompt: {session['seed_prompt']}",
            f"turn_count: {session['turn_count']}",
            "",
        ]
        for turn in session["turns"]:
            lines.append(f"--- turn {turn['turn_index']} ---")
            lines.append(f"input: {turn['input']}")
            lines.append(f"teacher_a: {turn['teacher_a']}")
            lines.append(f"teacher_b: {turn['teacher_b']}")
            lines.append(f"student: {turn['student']}")
            lines.append(f"winner: {turn['winner']}")
            lines.append(f"overall_scores: {json.dumps(turn['eval'].get('overall_scores', {}), ensure_ascii=False)}")
            lines.append(f"axis_scores: {json.dumps(turn['eval'].get('axis_scores', {}), ensure_ascii=False)}")
            lines.append(f"reason: {turn['eval']['reason']}")
            lines.append(f"chosen_text: {turn['chosen_text']}")
            if "rejected_text" in turn:
                lines.append(f"rejected_text: {turn['rejected_text']}")
            lines.append("")
        return "\n".join(lines)

    def _finalize_section(
        self,
        cycle_id: str,
        pending_turns: List[Dict[str, Any]],
        history: List[Dict[str, str]],
        turns: List[Dict[str, Any]],
    ) -> None:
        if not pending_turns:
            return

        eval_inputs = [
            {
                "prompt": t["input"],
                "teacher_a": t["teacher_a"],
                "teacher_b": t["teacher_b"],
                "student": t["student"],
            }
            for t in pending_turns
        ]
        eval_results = self.evaluator.evaluate_section(eval_inputs)

        for turn, eval_result in zip(pending_turns, eval_results):
            outputs = {
                "teacher_a": turn["teacher_a"],
                "teacher_b": turn["teacher_b"],
                "student": turn["student"],
            }
            winner = eval_result["winner"]
            chosen_text = outputs[winner]
            ranked = sorted(eval_result["overall_scores"].items(), key=lambda x: x[1], reverse=True)
            rejected_name = ranked[-1][0]
            rejected_text = outputs[rejected_name]

            turn["eval"] = eval_result
            turn["winner"] = winner
            turn["chosen_text"] = chosen_text
            turn["rejected_name"] = rejected_name
            turn["rejected_text"] = rejected_text
            turns.append(turn)

            LOGGER.info(
                f"[TURN_END] cycle_id={cycle_id} turn_index={turn['turn_index']} winner={winner} "
                f"rejected={rejected_name} overall={eval_result['overall_scores']} "
                f"chosen={short_text(chosen_text, 180)}"
            )

            live_block = (
                f"==============================\n"
                f"TURN {turn['turn_index']}\n"
                f"==============================\n\n"
                f"{self.teacher_a_label} >> {sanitize_live_view_text(outputs['teacher_a'])}\n\n"
                f"{self.teacher_b_label} >> {sanitize_live_view_text(outputs['teacher_b'])}\n\n"
                f"{self.student_label} >> {sanitize_live_view_text(outputs['student'])}\n\n"
                f"winner >> {winner}\n"
                f"rejected >> {rejected_name}"
            )
            append_live_view_block(live_block)

            history.append({"role": "assistant", "content": chosen_text})
            history.append({"role": "user", "content": self._make_next_prompt(chosen_text)})

    def _run_one_session(self, cycle_id: str) -> Dict[str, Any]:
        ttl = self.cfg.session_ttl
        seed_prompt = random.choice(SEED_PROMPTS)
        LOGGER.info(f"[SESSION_START] cycle_id={cycle_id} ttl={ttl} seed_prompt={seed_prompt}")

        history: List[Dict[str, str]] = [{"role": "user", "content": seed_prompt}]
        turns: List[Dict[str, Any]] = []
        pending_turns: List[Dict[str, Any]] = []

        while ttl > 0 and self.keep_running:
            turn_index = len(turns) + len(pending_turns)
            current_input = history[-1]["content"]
            LOGGER.info(
                f"[TURN_START] cycle_id={cycle_id} turn_index={turn_index} ttl={ttl} "
                f"input={short_text(current_input, 200)}"
            )

            teacher_order = [("teacher_a", self.teacher_a), ("teacher_b", self.teacher_b)]
            random.shuffle(teacher_order)

            teacher_outputs: Dict[str, str] = {}
            for name, model in teacher_order:
                teacher_outputs[name] = model.generate(
                    system_prompt=self.persona_text,
                    history=history[-self.cfg.persist_history_turns:],
                    max_new_tokens=self.cfg.max_new_tokens_teacher,
                    temperature=self.cfg.temperature_teacher,
                    top_p=self.cfg.top_p,
                    repetition_penalty=self.cfg.repetition_penalty,
                )

            student_text = self.student.generate(
                system_prompt=self.persona_text,
                history=history[-self.cfg.persist_history_turns:],
                max_new_tokens=self.cfg.max_new_tokens_student,
                temperature=self.cfg.temperature_student,
                top_p=self.cfg.top_p,
                repetition_penalty=self.cfg.repetition_penalty,
            )

            pending_turns.append({
                "turn_index": turn_index,
                "input": current_input,
                "teacher_a": teacher_outputs["teacher_a"],
                "teacher_b": teacher_outputs["teacher_b"],
                "student": student_text,
                "timestamp": now_ts(),
            })

            ttl -= 1
            if len(pending_turns) >= self.cfg.section_size:
                self._finalize_section(cycle_id, pending_turns, history, turns)
                pending_turns = []

        if pending_turns:
            self._finalize_section(cycle_id, pending_turns, history, turns)

        session = {
            "cycle_id": cycle_id,
            "seed_prompt": seed_prompt,
            "turn_count": len(turns),
            "turns": turns,
            "created_at": now_ts(),
            "student_model_path": self.student.current_model_path,
        }

        session_path = os.path.join(self.cfg.sessions_dir, f"{cycle_id}.json")
        conversation_text_path = os.path.join(self.cfg.conversations_dir, f"{cycle_id}.txt")
        self._submit_with_logging(cpu_flush_session_log, (session_path, session), f"session_json:{cycle_id}")
        self._submit_with_logging(
            cpu_flush_text,
            (conversation_text_path, self._render_conversation_text(session)),
            f"conversation_txt:{cycle_id}",
        )
        LOGGER.info(f"[SESSION_DONE] cycle_id={cycle_id} turn_count={len(turns)} session_path={session_path}")
        return session

    def _make_next_prompt(self, previous_answer: str) -> str:
        templates = [
            f"今の返答をもっと短くすると？ 前の返答: {previous_answer}",
            f"同じ意味で自然に言い換えると？ 前の返答: {previous_answer}",
            f"次に続けるなら短くどう返す？ 前の返答: {previous_answer}",
            f"要点だけ残して言い直すと？ 前の返答: {previous_answer}",
        ]
        return random.choice(templates)

    def _calc_sample_weight(self, winner: str, overall_scores: Dict[str, float]) -> float:
        ranked = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        winner_score = float(overall_scores.get(winner, 0.0))
        second_score = float(ranked[1][1]) if len(ranked) >= 2 else 0.0
        margin = max(0.0, winner_score - second_score)

        weight = 0.5 + winner_score * 1.5 + margin * self.cfg.margin_bonus_scale
        if winner == "student":
            weight += self.cfg.student_win_bonus

        weight = max(self.cfg.sample_weight_floor, min(self.cfg.sample_weight_ceiling, weight))
        return float(weight)

    def _build_datasets_from_sessions(
        self,
        batch_id: str,
        sessions: List[Dict[str, Any]],
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        total_turns = sum(len(session.get("turns", [])) for session in sessions)
        LOGGER.info(f"[DATASET_BUILD_START] batch_id={batch_id} session_count={len(sessions)} turn_count={total_turns}")

        sft_rows: List[Dict[str, Any]] = []
        dpo_rows: List[Dict[str, Any]] = []

        for session in sessions:
            cycle_id = session.get("cycle_id", "unknown_cycle")
            for turn in session["turns"]:
                winner = turn["winner"]
                overall_scores = turn["eval"].get("overall_scores", {})
                axis_scores = turn["eval"].get("axis_scores", {})
                winner_score = float(overall_scores.get(winner, 0.0))
                ranked = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
                second_name = ranked[1][0] if len(ranked) > 1 else None
                second_score = float(ranked[1][1]) if len(ranked) > 1 else 0.0
                margin = max(0.0, winner_score - second_score)
                sample_weight = self._calc_sample_weight(winner, overall_scores)

                if winner_score >= self.cfg.min_overall_to_train:
                    prompt = self._reconstruct_prompt_from_turn(turn["input"])
                    sft_rows.append({
                        "prompt": prompt,
                        "chosen": turn["chosen_text"],
                        "winner": winner,
                        "overall_scores": overall_scores,
                        "axis_scores": axis_scores,
                        "sample_weight": sample_weight,
                        "cycle_id": cycle_id,
                        "batch_id": batch_id,
                        "turn_index": turn["turn_index"],
                    })

                if (
                    self.cfg.dpo_enabled
                    and winner_score >= self.cfg.min_overall_to_train
                    and second_name is not None
                    and margin >= self.cfg.min_dpo_margin
                ):
                    dpo_rows.append({
                        "prompt": self._reconstruct_prompt_from_turn(turn["input"]),
                        "chosen": turn["chosen_text"],
                        "rejected": turn["rejected_text"],
                        "winner": winner,
                        "rejected_name": turn["rejected_name"],
                        "overall_scores": overall_scores,
                        "axis_scores": axis_scores,
                        "margin": margin,
                        "sample_weight": sample_weight,
                        "cycle_id": cycle_id,
                        "batch_id": batch_id,
                        "turn_index": turn["turn_index"],
                    })

        sft_path = None
        dpo_path = None

        if sft_rows:
            sft_path = os.path.join(self.cfg.dataset_dir, f"{batch_id}.jsonl")
            atomic_write_jsonl(sft_path, sft_rows)

        if dpo_rows:
            dpo_path = os.path.join(self.cfg.dpo_dataset_dir, f"{batch_id}.jsonl")
            atomic_write_jsonl(dpo_path, dpo_rows)

        best_source = self._dominant_winner(sft_rows) if sft_rows else None
        LOGGER.info(
            f"[DATASET_BUILD_DONE] batch_id={batch_id} sft_rows={len(sft_rows)} "
            f"dpo_rows={len(dpo_rows)} best_source={best_source}"
        )
        return sft_path, dpo_path, best_source

    def _dominant_winner(self, rows: List[Dict[str, Any]]) -> str:
        counter = {"teacher_a": 0, "teacher_b": 0, "student": 0}
        for r in rows:
            counter[r["winner"]] = counter.get(r["winner"], 0) + 1
        winner = sorted(counter.items(), key=lambda x: x[1], reverse=True)[0][0]
        LOGGER.info(f"[DOMINANT_WINNER] counter={counter} winner={winner}")
        return winner

    def _reconstruct_prompt_from_turn(self, user_input: str) -> str:
        return f"{self.persona_text}\n\nユーザー: {user_input}\nアシスタント: "


# =========================
# Entrypoint
# =========================
def main():
    cfg = AppConfig()
    LOGGER.info("[MAIN_START]")
    LOGGER.info(f"[BOOT_CONFIG_PRETTY]\n{json.dumps(asdict(cfg), ensure_ascii=False, indent=2)}")

    if not os.getenv("OPENAI_API_KEY"):
        LOGGER.warning("OPENAI_API_KEY is not set. OpenAI fallback will be unavailable.")
    if not _HAS_GOOGLE_GENAI:
        LOGGER.warning("google-genai package is not available. Gemini evaluator disabled.")
    if not _HAS_OPENAI:
        LOGGER.warning("openai package is not available. OpenAI fallback disabled.")
    if not os.getenv("GEMINI_API_KEY"):
        LOGGER.warning("Gemini primary evaluator may be unavailable without GEMINI_API_KEY.")

    runtime = EndlessTrainerRuntime(cfg)
    try:
        runtime.run()
    finally:
        runtime.shutdown()
        LOGGER.info("[MAIN_END]")


if __name__ == "__main__":
    main()