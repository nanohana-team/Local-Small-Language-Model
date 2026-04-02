import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import requests


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def simple_score(text: str) -> float:
    if not text:
        return 0.0

    score = 0.0
    length = len(text)

    if length < 10:
        score -= 1.0
    elif length <= 120:
        score += 2.0
    elif length <= 240:
        score += 0.5
    else:
        score -= 0.5

    if "。" in text:
        score += 1.0

    if "わからない" in text:
        score -= 1.0

    if any(marker in text for marker in ("Human:", "User:", "Assistant:", "AI:")):
        score -= 1.5

    return score


def llm_score(api_base: str, api_key: str, model: str, prompt: str, candidates: List[str]) -> List[int]:
    url = f"{api_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    eval_prompt = """
以下の応答候補を品質順にランキングしてください。

基準:
- 自然さ
- 意図適合
- 簡潔さ

入力:
{prompt}

候補:
""".format(prompt=prompt)

    for i, c in enumerate(candidates):
        eval_prompt += f"\n[{i}] {c}\n"

    eval_prompt += "\n順位をJSON配列で返してください。例: [2,0,1]"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": eval_prompt}],
        "temperature": 0.0,
    }

    res = requests.post(url, headers=headers, json=payload, timeout=120)
    res.raise_for_status()
    text = res.json()["choices"][0]["message"]["content"]

    try:
        ranking = json.loads(text)
    except Exception:
        ranking = list(range(len(candidates)))

    if not isinstance(ranking, list):
        ranking = list(range(len(candidates)))

    valid = []
    seen = set()
    for idx in ranking:
        if isinstance(idx, int) and 0 <= idx < len(candidates) and idx not in seen:
            valid.append(idx)
            seen.add(idx)
    for idx in range(len(candidates)):
        if idx not in seen:
            valid.append(idx)
    return valid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-file", required=True)
    p.add_argument("--output-file", required=True)
    p.add_argument("--use-llm", action="store_true")
    p.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    p.add_argument("--api-key", default="dummy")
    p.add_argument("--model", default="gpt-4o-mini")
    return p.parse_args()


def main():
    args = parse_args()

    data = load_jsonl(Path(args.input_file))
    grouped = defaultdict(list)
    for row in data:
        prompt_id = row.get("prompt_id")
        if prompt_id:
            grouped[prompt_id].append(row)

    output_rows = []

    for prompt_id, items in grouped.items():
        valid_items = []
        invalid_items = []

        for row in items:
            output = row.get("output")
            text = ""
            if isinstance(output, dict):
                text = str(output.get("text", "")).strip()

            if text:
                valid_items.append(row)
            else:
                row = dict(row)
                row["score"] = 0.0
                row["rank"] = 999999
                invalid_items.append(row)

        if valid_items:
            candidates = [x["output"]["text"] for x in valid_items]

            if args.use_llm:
                ranking = llm_score(
                    args.api_base,
                    args.api_key,
                    args.model,
                    str(valid_items[0].get("input", {}).get("text", "")),
                    candidates,
                )
            else:
                scores = [simple_score(c) for c in candidates]
                ranking = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

            for rank, idx in enumerate(ranking):
                item = dict(valid_items[idx])
                item["score"] = float(len(ranking) - rank)
                item["rank"] = rank
                output_rows.append(item)

        output_rows.extend(invalid_items)

    write_jsonl(Path(args.output_file), output_rows)
    print(f"[INFO] scored: {len(output_rows)}")


if __name__ == "__main__":
    main()
