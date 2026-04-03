
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
    GenerationConfig,
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
# DPO datasets
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
    teacher_model_id: str = "Qwen/Qwen2.5-7B-Instruct"

    # ---- CPU-targeted persona ----
    builtin_persona: str = """あなたは日本語で応答する軽量会話AIです。
目的は、CPUでも高速に動く短文対話モデルとして振る舞うことです。

ルール:
- 返答は原則1文、長くても2文まで
- 先に答えを述べる
- 回りくどい説明をしない
- 同じ内容を繰り返さない
- 聞かれたことに直接答える
- 不明なら短くそう言う
- 日本語だけで自然に返す
- 絵文字、顔文字、装飾記号は使わない
- 例や補足は必要な時だけ短く入れる
- CPU向け小型モデルとして、短く・自然に・破綻しにくく答える"""

    # ---- Generation ----
    max_new_tokens_teacher: int = 48
    max_new_tokens_student: int = 24
    teacher_temperature: float = 0.4
    student_temperature: float = 0.2
    top_p: float = 0.90
    repetition_penalty: float = 1.15
    generation_input_max_length: int = 1024

    # ---- Conversation loop ----
    session_ttl: int = 16
    persist_history_turns: int = 4
    section_size: int = 8
    sessions_per_eval: int = 2
    use_external_eval: bool = True
    endless: bool = True

    # ---- Training (CPU-targeted lightweight defaults) ----
    min_overall_to_train: float = 0.60
    min_dpo_margin: float = 0.05
    max_steps_per_cycle: int = 18
    train_batch_size: int = 1
    grad_accum: int = 4
    lr: float = 1.5e-4
    dpo_lr: float = 8e-6
    dpo_beta: float = 0.08
    dpo_loss_type: str = "sigmoid"
    dpo_enabled: bool = False
    lora_r: int = 4
    lora_alpha: int = 8
    lora_dropout: float = 0.03
    cutoff_len: int = 256
    save_every_cycle: bool = True

    # ---- CPU deployment profile ----
    export_cpu_profile: bool = True
    target_runtime_dtype: str = "float32"
    cpu_runtime_max_new_tokens: int = 40
    cpu_runtime_temperature: float = 0.2
    cpu_runtime_top_p: float = 0.9
    cpu_runtime_repetition_penalty: float = 1.10
    cpu_quantization_hint: str = "dynamic-int8-or-gguf-q4"

    # ---- Runtime ----
    seed: int = 42
    cpu_workers: int = max(1, (os.cpu_count() or 4) // 2)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else ("float16" if torch.cuda.is_available() else "float32")

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

    # ---- External eval compression / reduction ----
    external_eval_prefilter_enabled: bool = True
    external_eval_local_confidence_threshold: float = 0.30
    external_eval_cluster_enabled: bool = True
    external_eval_max_items_per_call: int = 3
    external_eval_summary_preview_chars: int = 72
    external_eval_summary_max_prompt_chars: int = 88
    external_eval_force_full_eval_every_n_sections: int = 0

    # ---- Eval axes ----
    eval_axes: Tuple[str, ...] = (
        "intent_fit",
        "naturalness",
        "consistency",
        "conciseness",
        "safety",
        "instruction_following",
        "language_purity",
        "robustness",
    )

    eval_axis_weights: Dict[str, float] = field(default_factory=lambda: {
        "intent_fit": 1.35,
        "naturalness": 1.35,
        "consistency": 1.00,
        "conciseness": 1.50,
        "safety": 1.25,
        "instruction_following": 1.40,
        "language_purity": 1.30,
        "robustness": 0.95,
    })

    sample_weight_floor: float = 0.25
    sample_weight_ceiling: float = 2.50
    margin_bonus_scale: float = 0.60
    student_win_bonus: float = 0.25


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


def is_gemma_family(model_id: str) -> bool:
    x = (model_id or "").lower()
    return "gemma" in x


def build_chat_messages(system_prompt: str, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    return messages


def render_fallback_prompt(model_id: str, system_prompt: str, history: List[Dict[str, str]], for_generation: bool = True) -> str:
    messages = build_chat_messages(system_prompt, history)

    if is_gemma_family(model_id):
        chunks: List[str] = []
        merged_system = system_prompt.strip()
        for msg in history:
            role = msg["role"]
            content = (msg["content"] or "").strip()
            if role == "user":
                user_text = content if not merged_system else f"{merged_system}\n\n{content}"
                chunks.append(f"<start_of_turn>user\n{user_text}<end_of_turn>")
                merged_system = ""
            elif role == "assistant":
                chunks.append(f"<start_of_turn>model\n{content}<end_of_turn>")
        if merged_system:
            chunks.append(f"<start_of_turn>user\n{merged_system}<end_of_turn>")
        if for_generation:
            chunks.append("<start_of_turn>model\n")
        return "\n".join(chunks)

    lines = [f"[SYSTEM]\n{system_prompt}\n"]
    for m in history:
        lines.append(f"[{m['role'].upper()}]\n{m['content']}\n")
    if for_generation:
        lines.append("[ASSISTANT]\n")
    return "\n".join(lines)


def clean_generated_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()

    stop_markers = [
        "\n[SYSTEM]", "\n[USER]", "\n[ASSISTANT]", "\n[MODEL]",
        "<start_of_turn>user", "<start_of_turn>model", "<end_of_turn>",
        "\nuser\n", "\nassistant\n", "\nmodel\n",
        "\n[PROMPT]", "\n[ANSWER]", "\n[REACTION]", "\n[DESCRIPTION]",
        "\n[HELP]", "\n[REPLAY]", "\n[MODELS]", "\n[LANGUAGE]",
        "\n[MODERATOR]", "\n[ADMINS]", "\n[CONTROLLER]",
    ]
    cut_positions = [text.find(marker) for marker in stop_markers if text.find(marker) >= 0]
    if cut_positions:
        text = text[:min(cut_positions)].strip()

    lines = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if s.startswith("[") and s.endswith("]"):
            break
        lines.append(s)

    text = "\n".join(lines).strip()

    # collapse repeated neighboring lines
    deduped = []
    prev = None
    repeat_count = 0
    for line in text.split("\n"):
        if line == prev:
            repeat_count += 1
            if repeat_count >= 2:
                continue
        else:
            repeat_count = 0
        deduped.append(line)
        prev = line
    text = "\n".join(deduped).strip()

    # keep only the first 1-2 sentences for CPU-oriented short replies
    sentence_parts = re.split(r"(?<=[。！？!?])\s*", text)
    sentence_parts = [x for x in sentence_parts if x]
    if len(sentence_parts) >= 2:
        text = "".join(sentence_parts[:2]).strip()

    return text[:240].strip()


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
def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(dtype_name, torch.float32)


class TextModel:
    def __init__(self, model_id: str, dtype: torch.dtype, max_input_length: int):
        self.model_id = model_id
        self.dtype = dtype
        self.max_input_length = max_input_length

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
        model_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "token": os.getenv("HF_TOKEN"),
        }
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "cuda"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        self.model.eval()
        LOGGER.info(
            f"[MODEL_LOAD_DONE] weights model={model_id} sec={time.perf_counter() - t1:.2f} "
            f"device={self.model.device}"
        )

    def _build_prompt(self, system_prompt: str, history: List[Dict[str, str]]) -> str:
        chat_template = getattr(self.tokenizer, "chat_template", None)
        if chat_template:
            try:
                return self.tokenizer.apply_chat_template(
                    build_chat_messages(system_prompt, history),
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as e:
                LOGGER.warning(f"[CHAT_TEMPLATE_FAIL] model={self.model_id} error={e}")

        return render_fallback_prompt(self.model_id, system_prompt, history, for_generation=True)

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
            max_length=self.max_input_length,
        ).to(self.model.device)
        input_tokens = int(inputs["input_ids"].shape[1])

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": 4,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if not generation_kwargs["do_sample"]:
            generation_kwargs.pop("temperature", None)
            generation_kwargs.pop("top_p", None)

        out = self.model.generate(**inputs, **generation_kwargs)
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        text = clean_generated_text(self.tokenizer.decode(gen_ids, skip_special_tokens=True))
        sec = time.perf_counter() - t0

        LOGGER.info(
            f"[GEN_DONE] model={self.model_id} in_tokens={input_tokens} out_tokens={len(gen_ids)} "
            f"out_chars={len(text)} sec={sec:.2f} text={short_text(text, 180)}"
        )
        return text


class StudentModel:
    def __init__(self, model_id: str, dtype: torch.dtype, max_input_length: int):
        self.base_model_id = model_id
        self.current_model_path = model_id
        self.dtype = dtype
        self.max_input_length = max_input_length
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

    def _from_pretrained_kwargs(self) -> Dict[str, Any]:
        kwargs = {
            "torch_dtype": self.dtype,
            "trust_remote_code": True,
            "token": os.getenv("HF_TOKEN"),
        }
        if torch.cuda.is_available():
            kwargs["device_map"] = "cuda"
        return kwargs

    def _load_model_from_path(self, model_path: str):
        LOGGER.info(f"[STUDENT_WEIGHTS_LOAD_START] model={model_path} dtype={self.dtype}")
        t0 = time.perf_counter()
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **self._from_pretrained_kwargs())
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
                return self.tokenizer.apply_chat_template(
                    build_chat_messages(system_prompt, history),
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as e:
                LOGGER.warning(f"[STUDENT_CHAT_TEMPLATE_FAIL] model={self.current_model_path} error={e}")

        return render_fallback_prompt(self.current_model_path, system_prompt, history, for_generation=True)

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
            max_length=self.max_input_length,
        ).to(self.model.device)
        input_tokens = int(inputs["input_ids"].shape[1])

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": 4,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if not generation_kwargs["do_sample"]:
            generation_kwargs.pop("temperature", None)
            generation_kwargs.pop("top_p", None)

        out = self.model.generate(**inputs, **generation_kwargs)
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        text = clean_generated_text(self.tokenizer.decode(gen_ids, skip_special_tokens=True))
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
        base_for_merge = AutoModelForCausalLM.from_pretrained(previous_model_path, **self._from_pretrained_kwargs())
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

            train_model = AutoModelForCausalLM.from_pretrained(previous_model_path, **self._from_pretrained_kwargs())
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
                bf16=torch.cuda.is_available() and self.dtype == torch.bfloat16,
                fp16=torch.cuda.is_available() and self.dtype == torch.float16,
                report_to=[],
                remove_unused_columns=False,
                optim="adamw_torch",
                lr_scheduler_type="cosine",
                warmup_ratio=0.05,
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

            policy_model = AutoModelForCausalLM.from_pretrained(previous_model_path, **self._from_pretrained_kwargs())
            ref_model = AutoModelForCausalLM.from_pretrained(previous_model_path, **self._from_pretrained_kwargs())

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
                bf16=torch.cuda.is_available() and self.dtype == torch.bfloat16,
                fp16=torch.cuda.is_available() and self.dtype == torch.float16,
                report_to=[],
                remove_unused_columns=False,
                optim="adamw_torch",
                lr_scheduler_type="cosine",
                warmup_ratio=0.05,
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
        self.section_counter = 0

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
                "first_prompt": short_text(items[0].get("prompt", ""), 180) if items else "",
                "first_teacher": short_text(items[0].get("teacher_preview", items[0].get("teacher", "")), 180) if items else "",
                "first_student": short_text(items[0].get("student_preview", items[0].get("student", "")), 180) if items else "",
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

    def _token_estimate(self, text: str) -> int:
        return max(1, (len(text or "") + 3) // 4)

    def _text_features(self, text: str, prompt: str = "") -> Dict[str, Any]:
        text = (text or "").strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        words = text.split()
        uniq_ratio = len(set(words)) / max(1, len(words))
        jp_chars = sum(1 for ch in text if "ぁ" <= ch <= "ヿ" or "一" <= ch <= "鿿")
        latin_chars = sum(1 for ch in text if ("a" <= ch.lower() <= "z"))
        punct_end = text.endswith(("。", "！", "？", "!", "?"))
        prompt_head = (prompt or "").strip()[:24]
        text_head = text[:24]
        features = {
            "chars": len(text),
            "lines": len(lines),
            "uniq_ratio": uniq_ratio,
            "jp_ratio": jp_chars / max(1, jp_chars + latin_chars),
            "has_role_tokens": any(tag in text for tag in ("[SYSTEM]", "[USER]", "[ASSISTANT]", "<start_of_turn>")),
            "has_meta_tokens": any(tag in text for tag in ("[PROMPT]", "[ANSWER]", "[HELP]", "[REPLAY]", "[MODELS]", "[DESCRIPTION]")),
            "has_prompt_echo": bool(prompt_head) and prompt_head in text_head,
            "repetitive": uniq_ratio < 0.45 or any(lines.count(ln) >= 2 for ln in set(lines) if ln),
            "truncated_like": not punct_end,
            "preview": text[: self.cfg.external_eval_summary_preview_chars],
        }
        return features

    def _compact_flags(self, features: Dict[str, Any]) -> str:
        flags = []
        if features["has_role_tokens"]:
            flags.append("role")
        if features["has_meta_tokens"]:
            flags.append("meta")
        if features["has_prompt_echo"]:
            flags.append("echo")
        if features["repetitive"]:
            flags.append("repeat")
        if features["truncated_like"]:
            flags.append("cut")
        if features["jp_ratio"] < 0.80:
            flags.append("langmix")
        return ",".join(flags) if flags else "clean"

    def _summarize_candidate(self, text: str, prompt: str = "") -> Tuple[str, Dict[str, Any]]:
        features = self._text_features(text, prompt)
        preview = short_text(features["preview"].replace("\n", " "), self.cfg.external_eval_summary_preview_chars)
        summary = (
            f"chars={features['chars']} | flags={self._compact_flags(features)} | "
            f"uniq={features['uniq_ratio']:.2f} | jp={features['jp_ratio']:.2f} | preview={preview}"
        )
        return summary, features

    def _local_fallback_eval(self, payload: Dict[str, str]) -> Dict[str, Any]:
        def score_text_axes(x: str, prompt: str) -> Dict[str, float]:
            text = (x or "").strip()
            feats = self._text_features(text, prompt)
            length = feats["chars"]
            naturalness = 0.58 + min(0.18, feats["uniq_ratio"] * 0.18)
            intent_fit = 0.70 if 4 <= length <= 88 else 0.46
            consistency = 0.62 if length >= 6 else 0.40
            conciseness = 0.96 if 4 <= length <= 64 else (0.78 if length <= 96 else 0.38)
            safety = 0.97
            instruction_following = 0.80 if 4 <= length <= 72 else 0.42
            language_purity = 0.96 if feats["jp_ratio"] >= 0.90 else (0.72 if feats["jp_ratio"] >= 0.70 else 0.35)
            robustness = 0.58

            if feats["has_role_tokens"]:
                naturalness -= 0.24
                instruction_following -= 0.24
                language_purity -= 0.15
            if feats["has_meta_tokens"]:
                naturalness -= 0.18
                instruction_following -= 0.20
            if feats["has_prompt_echo"]:
                intent_fit -= 0.12
                instruction_following -= 0.14
            if feats["repetitive"]:
                naturalness -= 0.20
                conciseness -= 0.18
                consistency -= 0.12
            if feats["truncated_like"] and length >= 32:
                naturalness -= 0.08
                consistency -= 0.08

            return {
                "intent_fit": float(max(0.0, min(1.0, intent_fit))),
                "naturalness": float(max(0.0, min(1.0, naturalness))),
                "consistency": float(max(0.0, min(1.0, consistency))),
                "conciseness": float(max(0.0, min(1.0, conciseness))),
                "safety": float(max(0.0, min(1.0, safety))),
                "instruction_following": float(max(0.0, min(1.0, instruction_following))),
                "language_purity": float(max(0.0, min(1.0, language_purity))),
                "robustness": float(max(0.0, min(1.0, robustness))),
            }

        prompt = payload.get("prompt", "")
        axis_scores = {
            "teacher": score_text_axes(payload["teacher"], prompt),
            "student": score_text_axes(payload["student"], prompt),
        }
        overall_scores = {k: self._weighted_mean(v) for k, v in axis_scores.items()}
        winner = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)[0][0]
        return {
            "winner": winner,
            "axis_scores": axis_scores,
            "overall_scores": overall_scores,
            "reason": "Local heuristic evaluation.",
        }

    def _estimate_local_confidence(self, payload: Dict[str, str], local_result: Dict[str, Any]) -> float:
        feats_teacher = self._text_features(payload.get("teacher", ""), payload.get("prompt", ""))
        feats_student = self._text_features(payload.get("student", ""), payload.get("prompt", ""))
        scores = local_result["overall_scores"]
        gap = abs(float(scores.get("teacher", 0.0)) - float(scores.get("student", 0.0)))
        confidence = gap
        if feats_student["has_role_tokens"] or feats_student["has_meta_tokens"]:
            confidence += 0.18
        if feats_student["repetitive"]:
            confidence += 0.12
        if feats_student["has_prompt_echo"]:
            confidence += 0.10
        if feats_teacher["chars"] <= 72 and not feats_teacher["has_role_tokens"] and not feats_teacher["repetitive"]:
            confidence += 0.06
        return float(max(0.0, min(1.0, confidence)))

    def _cluster_key(self, payload: Dict[str, str], local_result: Dict[str, Any]) -> Tuple[Any, ...]:
        tf = self._text_features(payload.get("teacher", ""), payload.get("prompt", ""))
        sf = self._text_features(payload.get("student", ""), payload.get("prompt", ""))
        gap = abs(float(local_result["overall_scores"].get("teacher", 0.0)) - float(local_result["overall_scores"].get("student", 0.0)))

        def bucket(n: int) -> str:
            if n <= 24:
                return "xs"
            if n <= 56:
                return "s"
            if n <= 96:
                return "m"
            return "l"

        return (
            bucket(tf["chars"]),
            bucket(sf["chars"]),
            tf["repetitive"],
            sf["repetitive"],
            sf["has_role_tokens"],
            sf["has_meta_tokens"],
            sf["has_prompt_echo"],
            round(gap, 1),
        )

    def _compress_item(self, turn_index: int, item: Dict[str, str], local_result: Dict[str, Any], confidence: float) -> Dict[str, Any]:
        teacher_summary, teacher_features = self._summarize_candidate(item["teacher"], item["prompt"])
        student_summary, student_features = self._summarize_candidate(item["student"], item["prompt"])
        prompt_short = short_text(item["prompt"], self.cfg.external_eval_summary_max_prompt_chars)
        return {
            "turn_index": turn_index,
            "prompt": prompt_short,
            "teacher": item["teacher"],
            "student": item["student"],
            "teacher_summary": teacher_summary,
            "student_summary": student_summary,
            "teacher_preview": teacher_features["preview"],
            "student_preview": student_features["preview"],
            "local_winner": local_result["winner"],
            "local_gap": abs(float(local_result["overall_scores"].get("teacher", 0.0)) - float(local_result["overall_scores"].get("student", 0.0))),
            "local_confidence": confidence,
        }

    def _safe_parse_section_eval_json(self, text: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
            for model_name in ("teacher", "student"):
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
                for name in ("teacher", "student")
            }
            winner = item.get("winner")
            if winner not in ("teacher", "student"):
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
                payload = {"prompt": items[i].get("prompt", ""), "teacher": items[i].get("teacher_preview", ""), "student": items[i].get("student_preview", "")}
                out.append(self._local_fallback_eval(payload))
            else:
                out.append(normalized[i])
        return out

    def _build_section_eval_prompt(self, items: List[Dict[str, Any]]) -> str:
        header = (
            "あなたは会話品質評価器です。JSONのみ返してください。\n"
            "各候補は要約済みです。teacher / student のうち、CPU向け短文日本語アシスタントとしてより良い方を選んでください。\n"
            "重視順: instruction_following, conciseness, naturalness, language_purity, intent_fit, consistency, robustness, safety。\n"
            "flagsの意味: role=役割トークン混入, meta=メタ記号混入, echo=プロンプト反復, repeat=繰り返し, cut=途中切れ, langmix=言語混入。\n"
            "要約のみを見て評価し、0.0000〜1.0000で採点してください。\n"
        )

        format_text = """
返却形式:
{
  \"results\": [
    {
      \"turn_index\": 0,
      \"winner\": \"teacher\" | \"student\",
      \"axis_scores\": {
        \"teacher\": {
          \"intent_fit\": 0.0000,
          \"naturalness\": 0.0000,
          \"consistency\": 0.0000,
          \"conciseness\": 0.0000,
          \"safety\": 0.0000,
          \"instruction_following\": 0.0000,
          \"language_purity\": 0.0000,
          \"robustness\": 0.0000
        },
        \"student\": {
          \"intent_fit\": 0.0000,
          \"naturalness\": 0.0000,
          \"consistency\": 0.0000,
          \"conciseness\": 0.0000,
          \"safety\": 0.0000,
          \"instruction_following\": 0.0000,
          \"language_purity\": 0.0000,
          \"robustness\": 0.0000
        }
      },
      \"reason\": \"短く\"
    }
  ]
}
""".strip()

        blocks = []
        for i, x in enumerate(items):
            blocks.append(
                f"TURN_INDEX: {i}\n"
                f"PROMPT: {x['prompt']}\n"
                f"TEACHER_SUMMARY: {x['teacher_summary']}\n"
                f"STUDENT_SUMMARY: {x['student_summary']}\n"
                f"LOCAL_HINT: winner={x['local_winner']} gap={x['local_gap']:.3f} confidence={x['local_confidence']:.3f}"
            )

        joined = "\n\n====================\n\n".join(blocks)
        prompt = f"{header}\n\n{format_text}\n\n評価対象:\n{joined}"
        LOGGER.info(
            f"[EVAL_PROMPT_COMPRESSED] items={len(items)} approx_tokens={self._token_estimate(prompt)}"
        )
        return prompt

    def _eval_section_with_gemini(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

    def _eval_section_with_openai(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

    def _expand_representative_result(self, base_result: Dict[str, Any], inherited_count: int) -> Dict[str, Any]:
        copied = json.loads(json.dumps(base_result, ensure_ascii=False))
        copied["reason"] = f"Representative eval propagated to {inherited_count} item(s). " + copied.get("reason", "")
        return copied

    def _run_external_eval(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not items:
            return []

        now = time.time()
        if self.cfg.use_external_eval:
            if self.gemini_client is not None:
                if now < self.gemini_disabled_until:
                    remaining = max(0, int(self.gemini_disabled_until - now))
                    LOGGER.info(f"[EVAL_GEMINI_SKIPPED] cooldown_remaining_sec={remaining}")
                else:
                    try:
                        self._log_eval_request_preview("Gemini", items)
                        results = self._eval_section_with_gemini(items)
                        LOGGER.info("[EVAL_SECTION_PROVIDER_SUCCESS] provider=Gemini")
                        return results
                    except Exception as e:
                        msg = str(e)
                        if self._is_quota_error(msg):
                            retry_sec = self._extract_retry_delay_sec(msg)
                            cooldown_sec = retry_sec + 5 if retry_sec > 0 else min(self.cfg.gemini_cooldown_sec, 300)
                            self.gemini_disabled_until = time.time() + cooldown_sec
                            LOGGER.warning(f"[EVAL_SECTION_FAIL][GeminiQuota] cooldown_sec={cooldown_sec} message={msg}")
                        else:
                            LOGGER.warning(f"[EVAL_SECTION_FAIL][Gemini] {msg}", exc_info=True)

            if self.openai_client is not None:
                try:
                    self._log_eval_request_preview("OpenAI", items)
                    results = self._eval_section_with_openai(items)
                    LOGGER.info("[EVAL_SECTION_PROVIDER_SUCCESS] provider=OpenAI")
                    return results
                except Exception as e:
                    LOGGER.warning(f"[EVAL_SECTION_FAIL][OpenAI] {e}", exc_info=True)

        LOGGER.info("[EVAL_SECTION_FALLBACK] using local heuristic evaluator")
        results = [self._local_fallback_eval(x) for x in items]
        self._log_eval_response_summary("LocalFallback", results)
        return results

    def evaluate_section(self, items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        if not items:
            return []

        self.section_counter += 1
        LOGGER.info(f"[EVAL_SECTION_START] count={len(items)} section_counter={self.section_counter}")

        local_results = [self._local_fallback_eval(x) for x in items]
        final_results: List[Optional[Dict[str, Any]]] = [None] * len(items)
        uncertain_records: List[Dict[str, Any]] = []

        force_full_eval = (
            self.cfg.external_eval_force_full_eval_every_n_sections > 0
            and self.section_counter % self.cfg.external_eval_force_full_eval_every_n_sections == 0
        )

        for idx, (item, local_result) in enumerate(zip(items, local_results)):
            confidence = self._estimate_local_confidence(item, local_result)
            if self.cfg.external_eval_prefilter_enabled and not force_full_eval and confidence >= self.cfg.external_eval_local_confidence_threshold:
                local_copy = json.loads(json.dumps(local_result, ensure_ascii=False))
                local_copy["reason"] = f"High-confidence local prefilter (confidence={confidence:.3f}). " + local_copy.get("reason", "")
                final_results[idx] = local_copy
            else:
                uncertain_records.append({
                    "orig_index": idx,
                    "item": item,
                    "local_result": local_result,
                    "confidence": confidence,
                    "cluster_key": self._cluster_key(item, local_result),
                })

        if not uncertain_records:
            LOGGER.info("[EVAL_REDUCTION] all_items_prefiltered=true")
            return [x for x in final_results if x is not None]

        clusters: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
        if self.cfg.external_eval_cluster_enabled and not force_full_eval:
            for rec in uncertain_records:
                clusters.setdefault(rec["cluster_key"], []).append(rec)
        else:
            for rec in uncertain_records:
                clusters[(rec["orig_index"],)] = [rec]

        representatives = []
        for key, group in clusters.items():
            rep = sorted(group, key=lambda r: (r["confidence"], r["orig_index"]))[0]
            representatives.append((key, rep, group))

        representatives.sort(key=lambda x: (x[1]["confidence"], x[1]["orig_index"]))
        budget = max(1, self.cfg.external_eval_max_items_per_call)
        selected = representatives[:budget]
        skipped = representatives[budget:]

        compressed_items = [
            self._compress_item(rep["orig_index"], rep["item"], rep["local_result"], rep["confidence"])
            for _, rep, _ in selected
        ]

        before_est = sum(self._token_estimate(it["prompt"] + it["teacher"] + it["student"]) for it in items)
        after_est = sum(self._token_estimate(self._build_section_eval_prompt([ci])) for ci in compressed_items) if compressed_items else 0
        LOGGER.info(
            f"[EVAL_REDUCTION] total={len(items)} prefiltered={sum(x is not None for x in final_results)} "
            f"clusters={len(clusters)} selected_reps={len(selected)} skipped_clusters={len(skipped)} "
            f"approx_tokens_before={before_est} approx_tokens_after={after_est}"
        )

        external_results = self._run_external_eval(compressed_items) if compressed_items else []

        for (_, rep, group), rep_result in zip(selected, external_results):
            expanded = self._expand_representative_result(rep_result, len(group))
            for member in group:
                final_results[member["orig_index"]] = json.loads(json.dumps(expanded, ensure_ascii=False))

        for _, _, group in skipped:
            for member in group:
                local_copy = json.loads(json.dumps(member["local_result"], ensure_ascii=False))
                local_copy["reason"] = (
                    f"Representative skipped by external-eval budget; local result used (confidence={member['confidence']:.3f}). "
                    + local_copy.get("reason", "")
                )
                final_results[member["orig_index"]] = local_copy

        out = [x for x in final_results if x is not None]
        if len(out) != len(items):
            raise RuntimeError(f"evaluate_section produced mismatched result count: got={len(out)} expected={len(items)}")
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
            LOGGER.warning("CUDA is not available. Training will run on CPU and be slow.")

        dtype = resolve_torch_dtype(cfg.dtype)

        LOGGER.info(f"[BOOT] Loading teacher on GPU/CPU: {cfg.teacher_model_id}")
        self.teacher = TextModel(
            cfg.teacher_model_id,
            dtype=dtype,
            max_input_length=cfg.generation_input_max_length,
        )

        LOGGER.info(f"[BOOT] Loading student on GPU/CPU: {cfg.student_model_id}")
        self.student = StudentModel(
            cfg.student_model_id,
            dtype=dtype,
            max_input_length=cfg.generation_input_max_length,
        )

        LOGGER.info("[BOOT] Initializing external evaluator")
        self.evaluator = ExternalEvaluator(cfg)

        LOGGER.info(f"[BOOT] Starting CPU worker pool: workers={cfg.cpu_workers}")
        self.cpu_pool = ProcessPoolExecutor(max_workers=cfg.cpu_workers)
        self.cycle_index = 0
        self.keep_running = True
        self.teacher_label = format_model_label(cfg.teacher_model_id)
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
                        if self.cfg.export_cpu_profile:
                            self._export_cpu_runtime_profile(merged_after_sft)

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
                        if self.cfg.export_cpu_profile:
                            self._export_cpu_runtime_profile(merged_dir)
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

    def _export_cpu_runtime_profile(self, model_dir: str) -> None:
        profile = {
            "target": "cpu",
            "recommended_dtype": self.cfg.target_runtime_dtype,
            "quantization_hint": self.cfg.cpu_quantization_hint,
            "max_new_tokens": self.cfg.cpu_runtime_max_new_tokens,
            "temperature": self.cfg.cpu_runtime_temperature,
            "top_p": self.cfg.cpu_runtime_top_p,
            "repetition_penalty": self.cfg.cpu_runtime_repetition_penalty,
            "max_input_length": self.cfg.generation_input_max_length,
            "notes": [
                "single-teacher distilled model",
                "trained for short Japanese responses",
                "prefer dynamic int8 or gguf q4/q5 for CPU deployment",
                "keep prompt short and avoid long histories",
            ],
        }
        atomic_write_json(os.path.join(model_dir, "cpu_runtime_profile.json"), profile)

        generation_config = GenerationConfig(
            max_new_tokens=self.cfg.cpu_runtime_max_new_tokens,
            temperature=self.cfg.cpu_runtime_temperature,
            top_p=self.cfg.cpu_runtime_top_p,
            repetition_penalty=self.cfg.cpu_runtime_repetition_penalty,
            do_sample=self.cfg.cpu_runtime_temperature > 0,
            pad_token_id=self.student.tokenizer.pad_token_id,
            eos_token_id=self.student.tokenizer.eos_token_id,
        )
        generation_config.save_pretrained(model_dir)
        LOGGER.info(f"[CPU_PROFILE_EXPORTED] model_dir={model_dir}")

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
            lines.append(f"teacher: {turn['teacher']}")
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
                "teacher": t["teacher"],
                "student": t["student"],
            }
            for t in pending_turns
        ]
        eval_results = self.evaluator.evaluate_section(eval_inputs)

        for turn, eval_result in zip(pending_turns, eval_results):
            outputs = {
                "teacher": turn["teacher"],
                "student": turn["student"],
            }
            winner = eval_result["winner"]
            chosen_text = outputs[winner]
            rejected_name = "student" if winner == "teacher" else "teacher"
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
                f"{self.teacher_label} >> {sanitize_live_view_text(outputs['teacher'])}\n\n"
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

            teacher_text = self.teacher.generate(
                system_prompt=self.persona_text,
                history=history[-self.cfg.persist_history_turns:],
                max_new_tokens=self.cfg.max_new_tokens_teacher,
                temperature=self.cfg.teacher_temperature,
                top_p=self.cfg.top_p,
                repetition_penalty=self.cfg.repetition_penalty,
            )

            student_text = self.student.generate(
                system_prompt=self.persona_text,
                history=history[-self.cfg.persist_history_turns:],
                max_new_tokens=self.cfg.max_new_tokens_student,
                temperature=self.cfg.student_temperature,
                top_p=self.cfg.top_p,
                repetition_penalty=self.cfg.repetition_penalty,
            )

            pending_turns.append({
                "turn_index": turn_index,
                "input": current_input,
                "teacher": teacher_text,
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
            "teacher_model_id": self.cfg.teacher_model_id,
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

        weight = 0.5 + winner_score * 1.2 + margin * self.cfg.margin_bonus_scale
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
        counter = {"teacher": 0, "student": 0}
        for r in rows:
            counter[r["winner"]] = counter.get(r["winner"], 0) + 1
        winner = sorted(counter.items(), key=lambda x: x[1], reverse=True)[0][0]
        LOGGER.info(f"[DOMINANT_WINNER] counter={counter} winner={winner}")
        return winner

    def _reconstruct_prompt_from_turn(self, user_input: str) -> str:
        history = [{"role": "user", "content": user_input}]
        return render_fallback_prompt(self.cfg.student_model_id, self.persona_text, history, for_generation=True)


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
