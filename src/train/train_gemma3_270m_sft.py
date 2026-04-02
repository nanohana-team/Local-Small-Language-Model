import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


SYSTEM_FALLBACK = (
    "あなたは日本語で自然に会話する小型AIです。"
    "簡潔で自然な返答を心がけてください。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Gemma 3 270M with LoRA on JSONL chat data."
    )
    parser.add_argument("--model-name", type=str, default="google/gemma-3-270m-it", help="Base model or adapter path.")
    parser.add_argument("--train-file", type=str, required=True, help="Path to training JSONL file.")
    parser.add_argument("--eval-file", type=str, default="", help="Optional path to evaluation JSONL file.")
    parser.add_argument("--output-dir", type=str, default="outputs/gemma3-270m-lora", help="Directory to save LoRA adapter and checkpoints.")
    parser.add_argument("--merged-output-dir", type=str, default="outputs/gemma3-270m-merged", help="Directory to save merged full model after training.")
    parser.add_argument("--max-length", type=int, default=512, help="Max token length.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=8, help="Train batch size per device.")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8, help="Eval batch size per device.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--num-train-epochs", type=float, default=3.0, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio.")
    parser.add_argument("--logging-steps", type=int, default=10, help="Logging interval.")
    parser.add_argument("--save-steps", type=int, default=200, help="Checkpoint save interval.")
    parser.add_argument("--eval-steps", type=int, default=200, help="Eval interval when eval file is provided.")
    parser.add_argument("--save-total-limit", type=int, default=2, help="Max number of checkpoints to keep.")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 if supported.")
    parser.add_argument("--fp16", action="store_true", help="Use float16 if supported.")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing.")
    parser.add_argument("--merge-after-training", action="store_true", help="Merge LoRA adapter into base model after training.")
    return parser.parse_args()


def ensure_pad_token(tokenizer: AutoTokenizer, model: AutoModelForCausalLM | None = None) -> None:
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            if model is not None:
                model.resize_token_embeddings(len(tokenizer))


def normalize_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
    if "messages" in example and isinstance(example["messages"], list):
        msgs = []
        for m in example["messages"]:
            role = str(m.get("role", "")).strip()
            content = str(m.get("content", "")).strip()
            if role and content:
                msgs.append({"role": role, "content": content})
        if msgs:
            return msgs

    if "instruction" in example and "response" in example:
        return [
            {"role": "system", "content": SYSTEM_FALLBACK},
            {"role": "user", "content": str(example["instruction"]).strip()},
            {"role": "assistant", "content": str(example["response"]).strip()},
        ]

    if "input" in example and "output" in example:
        return [
            {"role": "system", "content": SYSTEM_FALLBACK},
            {"role": "user", "content": str(example["input"]).strip()},
            {"role": "assistant", "content": str(example["output"]).strip()},
        ]

    raise ValueError(
        "Unsupported record format. Each line must contain either "
        "'messages', or ('instruction','response'), or ('input','output')."
    )


def build_prompt_text(tokenizer: AutoTokenizer, messages: List[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def tokenize_example(example: Dict[str, Any], tokenizer: AutoTokenizer, max_length: int) -> Dict[str, Any]:
    messages = normalize_messages(example)
    text = build_prompt_text(tokenizer, messages)
    tokenized = tokenizer(text, truncation=True, max_length=max_length, padding=False)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def load_jsonl_dataset(path: str) -> Dataset:
    return load_dataset("json", data_files=path, split="train")


def get_torch_dtype(args: argparse.Namespace) -> torch.dtype:
    if args.bf16 and args.fp16:
        raise ValueError("bf16 and fp16 cannot both be enabled.")
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return torch.float32


def resolve_base_model_name(model_name_or_path: str) -> tuple[str, bool]:
    model_path = Path(model_name_or_path)
    adapter_config_path = model_path / "adapter_config.json"
    if adapter_config_path.exists():
        peft_cfg = PeftConfig.from_pretrained(model_name_or_path)
        return str(peft_cfg.base_model_name_or_path), True
    return model_name_or_path, False


def build_model_and_tokenizer(args: argparse.Namespace):
    torch_dtype = get_torch_dtype(args)
    base_model_name, is_adapter_dir = resolve_base_model_name(args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
    )

    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
        base_model.config.use_cache = False

    ensure_pad_token(tokenizer, base_model)

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    if is_adapter_dir:
        print(f"[INFO] Resuming from LoRA adapter: {args.model_name}")
        model = PeftModel.from_pretrained(
            base_model,
            args.model_name,
            is_trainable=True,
        )
    else:
        print(f"[INFO] Starting from base model: {args.model_name}")
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(base_model, peft_config)

    model.print_trainable_parameters()
    return model, tokenizer, base_model_name


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.merged_output_dir).parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer, base_model_name = build_model_and_tokenizer(args)

    train_dataset = load_jsonl_dataset(args.train_file)
    train_dataset = train_dataset.map(
        lambda x: tokenize_example(x, tokenizer, args.max_length),
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset",
    )

    eval_dataset = None
    if args.eval_file and Path(args.eval_file).exists() and Path(args.eval_file).stat().st_size > 0:
        eval_dataset = load_jsonl_dataset(args.eval_file)
        if len(eval_dataset) > 0:
            eval_dataset = eval_dataset.map(
                lambda x: tokenize_example(x, tokenizer, args.max_length),
                remove_columns=eval_dataset.column_names,
                desc="Tokenizing eval dataset",
            )
        else:
            eval_dataset = None

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    evaluation_strategy = "steps" if eval_dataset is not None else "no"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        eval_strategy=evaluation_strategy,
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        dataloader_pin_memory=torch.cuda.is_available(),
        report_to="none",
        seed=args.seed,
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"[INFO] LoRA adapter saved to: {args.output_dir}")

    if args.merge_after_training:
        print("[INFO] Merging LoRA adapter into base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=get_torch_dtype(args),
        )
        merged_model = PeftModel.from_pretrained(base_model, args.output_dir)
        merged_model = merged_model.merge_and_unload()

        Path(args.merged_output_dir).mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(args.merged_output_dir)
        tokenizer.save_pretrained(args.merged_output_dir)
        print(f"[INFO] Merged model saved to: {args.merged_output_dir}")


if __name__ == "__main__":
    main()
