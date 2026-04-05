from __future__ import annotations

import argparse
import json
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.apps.repeat_divergence_convergence import run_divergence_convergence
from src.llm.response_llm_api import generate_response


def clamp(x: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, x))


def unique_preserve(items: List[str]) -> List[str]:
    return list(dict.fromkeys(items))


def load_prompt_template(path: str = "settings/llm_api_prompt.txt") -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def compute_diversity(words: List[str]) -> float:
    if not words:
        return 0.0
    return len(set(words)) / (len(words) + 2)


def compute_convergence(input_words: List[str], final_words: List[str]) -> float:
    if not final_words:
        return 0.0
    overlap = len(set(input_words) & set(final_words))
    return overlap / len(final_words)


def compute_reward(
    quality_score: float,
    diversity_score: float,
    convergence_score: float,
    iteration_count: int,
    relevance_score: float = 0.0,
    coherence_score: float = 0.0,
    focus_score: float = 0.0,
    noise_score: float = 0.0,
) -> float:
    # 低品質は強制的に負報酬
    if quality_score < 0.20:
        penalty = -1.0 - 0.2 * max(0, iteration_count - 1)
        penalty -= 0.5 * max(0.0, 0.2 - quality_score)
        return penalty

    base_reward = (
        2.0 * quality_score
        + 0.25 * diversity_score
        + 0.8 * convergence_score
        - 0.3 * iteration_count
    )

    # 関連性・一貫性・収束性を明示的に加点
    structure_bonus = (
        1.0 * relevance_score
        + 0.5 * coherence_score
        + 0.5 * focus_score
        - 0.5 * (1.0 - noise_score)
    )

    # 品質×収束 の相乗ボーナス
    semantic_bonus = 1.5 * quality_score * convergence_score

    return base_reward + structure_bonus + semantic_bonus


_FLOAT_PATTERN = re.compile(r"(?<!\d)(0(?:\.\d+)?|1(?:\.0+)?)(?!\d)")


def _extract_float_scores(text: str, expected_count: int) -> List[float]:
    scores: List[float] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _FLOAT_PATTERN.search(line)
        if m:
            try:
                scores.append(clamp(float(m.group(1))))
            except Exception:
                pass

    if len(scores) < expected_count:
        found = _FLOAT_PATTERN.findall(text)
        fallback_scores: List[float] = []
        for item in found:
            try:
                fallback_scores.append(clamp(float(item)))
            except Exception:
                pass
        scores = fallback_scores[:expected_count]

    return scores[:expected_count]


def _extract_json_object_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in LLM response")
    return stripped[start:end + 1]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return clamp(float(value))
    except Exception:
        return default


def _normalize_eval_result(data: Dict[str, Any]) -> Dict[str, Any]:
    relevance = _safe_float(data.get("relevance", 0.0))
    coherence = _safe_float(data.get("coherence", 0.0))
    focus = _safe_float(data.get("focus", 0.0))
    noise = _safe_float(data.get("noise", 0.0))
    score = _safe_float(data.get("score", 0.0))

    issues = data.get("issues", [])
    if not isinstance(issues, list):
        issues = []
    issues = [str(x) for x in issues[:8]]

    improvements = data.get("improvements", [])
    if not isinstance(improvements, list):
        improvements = []
    improvements = [str(x) for x in improvements[:8]]

    return {
        "relevance": relevance,
        "coherence": coherence,
        "focus": focus,
        "noise": noise,
        "score": score,
        "issues": issues,
        "improvements": improvements,
    }


def _build_single_eval_prompt(record: Dict[str, Any]) -> str:
    template = load_prompt_template()
    return (
        template
        + "\n\n"
        + "あなたは評価専用AIです。\n"
        + "以下の入力と出力の関係を厳密に評価してください。\n\n"
        + "内部評価項目:\n"
        + "- relevance: 入力との意味的関連性\n"
        + "- coherence: 出力語同士のまとまり\n"
        + "- focus: 入力から適切に絞られているか\n"
        + "- noise: 不自然語や意味不明語の少なさ\n\n"
        + "score の計算式:\n"
        + "score = 0.4 * relevance + 0.3 * coherence + 0.2 * focus + 0.1 * noise\n\n"
        + "必須条件:\n"
        + "- JSONオブジェクトのみ返す\n"
        + "- explanation文は禁止\n"
        + "- markdown禁止\n"
        + '- 形式は {"relevance":0.0,"coherence":0.0,"focus":0.0,"noise":0.0,"score":0.0,"issues":["..."],"improvements":["..."]}\n'
        + "- issues は問題点を短く列挙\n"
        + "- improvements は改善提案を短く列挙\n\n"
        + f"input: {' '.join(record['input'])}\n"
        + f"output: {' '.join(record['final_words'])}\n"
        + f"iterations: {record.get('iterations', 0)}\n"
        + f"top_k: {record.get('top_k', 0)}\n"
        + f"top_n: {record.get('top_n', 0)}\n"
        + f"bias: {' '.join(record.get('bias', []))}\n"
    )


def llm_eval(record: Dict[str, Any]) -> Dict[str, Any]:
    prompt = _build_single_eval_prompt(record)
    try:
        response = generate_response(prompt)
        print("[LLM EVAL RAW RESPONSE]")
        print(response)

        json_text = _extract_json_object_text(response)
        parsed = json.loads(json_text)
        if not isinstance(parsed, dict):
            raise ValueError("LLM eval response must be a JSON object")

        result = _normalize_eval_result(parsed)
        print("[LLM EVAL PARSED]")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    except Exception as e:
        print(f"[LLM EVAL ERROR] {e}")
        return {
            "relevance": 0.0,
            "coherence": 0.0,
            "focus": 0.0,
            "noise": 0.0,
            "score": 0.0,
            "issues": ["evaluation_failed"],
            "improvements": [],
        }


def batch_llm_eval(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not records:
        return []
    return [llm_eval(r) for r in records]


class AxisPolicy:
    def __init__(self, num_axes: int):
        self.num_axes = num_axes
        self.weights = [random.uniform(0.3, 0.7) for _ in range(num_axes)]

    def sample(self) -> List[float]:
        return [clamp(w + random.uniform(-0.1, 0.1), 0.0, 1.0) for w in self.weights]

    def update(self, action: List[float], reward: float, lr: float = 0.05):
        for i in range(self.num_axes):
            self.weights[i] += lr * reward * (action[i] - self.weights[i])
            self.weights[i] = clamp(self.weights[i], 0.0, 1.0)


def save_learning_log(record: Dict[str, Any], policy_weights: List[float]):
    base_dir = Path("learned")
    logs_dir = base_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    filename = now.strftime("%Y_%m_%d_%H_%M_%S")

    counter = 0
    while True:
        path = logs_dir / f"{filename}_{counter}.json"
        if not path.exists():
            break
        counter += 1

    data = {"record": record, "policy_weights": policy_weights}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    latest_path = base_dir / "latest.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_latest_weights() -> Optional[List[float]]:
    path = Path("learned/latest.json")
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        weights = data.get("policy_weights")
        return weights if isinstance(weights, list) else None
    except Exception:
        return None


def compute_top_k(axis_weights: List[float]) -> int:
    semantic_idx = 1 if len(axis_weights) > 1 else 0
    abstraction_idx = 4 if len(axis_weights) > 4 else semantic_idx
    strength = (axis_weights[semantic_idx] + axis_weights[abstraction_idx]) / 2
    return int(3 + strength * 7)


def compute_top_n(axis_weights: List[float]) -> int:
    social_idx = 8 if len(axis_weights) > 8 else len(axis_weights) - 1
    formality_idx = 9 if len(axis_weights) > 9 else social_idx
    focus = (axis_weights[formality_idx] + axis_weights[social_idx]) / 2
    return int(4 + (1 - focus) * 8)


def compute_bias(axis_weights: List[float], axes: List[str]) -> List[str]:
    bias = []
    limit = min(len(axis_weights), len(axes))
    for i in range(limit):
        w = axis_weights[i]
        if w > 0.6:
            bias.append(f"{axes[i]}=0.2")
        elif w < 0.4:
            bias.append(f"{axes[i]}=-0.2")
    return bias


class Learner:
    def __init__(self, lexicon_path: str = "libs/dict_30000_hierarchical.json"):
        tmp = run_divergence_convergence(words=["私"], lexicon_path=lexicon_path, summary_only=True)
        self.axes = tmp["axes"]
        self.policy = AxisPolicy(num_axes=len(self.axes))
        self.history: List[Dict[str, Any]] = []
        self.lexicon_path = lexicon_path

        loaded = load_latest_weights()
        if loaded and len(loaded) == len(self.axes):
            print("[LOAD] Loaded previous policy weights")
            self.policy.weights = loaded

    def run_episode(self, input_words: List[str]) -> Dict[str, Any]:
        axis_weights = self.policy.sample()
        top_k = compute_top_k(axis_weights)
        top_n = compute_top_n(axis_weights)
        bias = compute_bias(axis_weights, self.axes)

        result = run_divergence_convergence(
            words=input_words,
            lexicon_path=self.lexicon_path,
            axis_weights=axis_weights,
            top_k=top_k,
            top_n=top_n,
            bias=bias,
            summary_only=True,
        )

        final_words = result.get("final_words", [])
        iterations = result.get("iterations", [])

        diversity_score = compute_diversity(final_words)
        convergence_score = compute_convergence(input_words, final_words)

        record = {
            "input": input_words,
            "axis_weights": axis_weights,
            "final_words": final_words,
            "diversity": diversity_score,
            "convergence": convergence_score,
            "iterations": len(iterations),
            "top_k": top_k,
            "top_n": top_n,
            "bias": bias,
        }
        self.history.append(record)
        return record

    def evaluate_and_learn(self, record: Dict[str, Any]) -> Dict[str, Any]:
        eval_result = llm_eval(record)

        quality = eval_result["score"]
        relevance = eval_result["relevance"]
        coherence = eval_result["coherence"]
        focus = eval_result["focus"]
        noise = eval_result["noise"]

        reward = compute_reward(
            quality_score=quality,
            diversity_score=record["diversity"],
            convergence_score=record["convergence"],
            iteration_count=record["iterations"],
            relevance_score=relevance,
            coherence_score=coherence,
            focus_score=focus,
            noise_score=noise,
        )

        self.policy.update(record["axis_weights"], reward)

        record["quality"] = quality
        record["reward"] = reward
        record["eval"] = eval_result

        save_learning_log(record, self.policy.weights)
        return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--words", nargs="*", default=["私", "は", "君", "です"])
    parser.add_argument("--lexicon", type=str, default="libs/dict_30000_hierarchical.json")
    args = parser.parse_args()

    learner = Learner(lexicon_path=args.lexicon)

    for i in range(args.episodes):
        print(f"\n=== EPISODE {i + 1} ===")
        r = learner.run_episode(args.words)
        print("[EPISODE RESULT]")
        print(json.dumps(r, ensure_ascii=False, indent=2))

        r = learner.evaluate_and_learn(r)
        print("[LEARN RESULT]")
        print(json.dumps({
            "quality": r["quality"],
            "reward": r["reward"],
            "eval": r["eval"],
        }, ensure_ascii=False, indent=2))