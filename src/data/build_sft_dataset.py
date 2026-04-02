import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-file", required=True)
    p.add_argument("--train-output-file", required=True)
    p.add_argument("--eval-output-file", required=True)
    p.add_argument("--eval-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def is_valid_ranked_row(row: Dict[str, Any]) -> bool:
    if "rank" not in row:
        return False
    output = row.get("output")
    if not isinstance(output, dict):
        return False
    if not str(output.get("text", "")).strip():
        return False
    input_obj = row.get("input")
    if not isinstance(input_obj, dict):
        return False
    return bool(str(input_obj.get("text", "")).strip())


def main():
    args = parse_args()
    random.seed(args.seed)

    data = load_jsonl(Path(args.input_file))

    grouped = defaultdict(list)
    for row in data:
        prompt_id = row.get("prompt_id")
        if prompt_id:
            grouped[prompt_id].append(row)

    sft_data: List[Dict[str, Any]] = []

    for prompt_id, items in grouped.items():
        valid_items = [row for row in items if is_valid_ranked_row(row)]
        if not valid_items:
            continue

        best = min(valid_items, key=lambda x: int(x["rank"]))
        sft_data.append(
            {
                "messages": [
                    {"role": "system", "content": "あなたは自然な日本語で会話するAIです。"},
                    {"role": "user", "content": best["input"]["text"]},
                    {"role": "assistant", "content": best["output"]["text"]},
                ]
            }
        )

    random.shuffle(sft_data)

    total = len(sft_data)
    if total == 0:
        train, eval_ = [], []
    elif total == 1:
        train, eval_ = sft_data, []
    else:
        split = int(total * (1 - args.eval_ratio))
        split = max(1, min(split, total - 1))
        train = sft_data[:split]
        eval_ = sft_data[split:]

    write_jsonl(Path(args.train_output_file), train)
    write_jsonl(Path(args.eval_output_file), eval_)

    print(f"[INFO] total={len(sft_data)} train={len(train)} eval={len(eval_)}")


if __name__ == "__main__":
    main()
