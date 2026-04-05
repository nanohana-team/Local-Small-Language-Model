from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# =========================
# データ構造
# =========================

@dataclass
class Candidate:
    diverged: List[str]
    converged: List[str]
    response: str


@dataclass
class Evaluation:
    divergence_quality: float
    convergence_quality: float
    response_naturalness: float
    thinking_efficiency: float
    overall_score: float


# =========================
# 発散モデル
# =========================

class DivergenceModel:
    def __init__(self, vocabulary: List[str], random_seed: int | None = None) -> None:
        self.vocabulary = vocabulary
        self.rng = random.Random(random_seed)

        # tokenごとの発散しやすさ重み
        self.token_weights: Dict[str, float] = {token: 0.0 for token in vocabulary}

    def propose(self, state_tokens: List[str], top_k: int = 5) -> List[str]:
        # 状態に近そうな単語 + 重み付きで候補生成
        scores: List[Tuple[str, float]] = []
        state_set = set(state_tokens)

        for token in self.vocabulary:
            score = self.token_weights.get(token, 0.0)

            # 単純な関連性ヒューリスティック
            if token in state_set:
                score += 0.4
            if any(token.startswith(s[:1]) for s in state_tokens if s):
                score += 0.1

            # 少し探索性を持たせる
            score += self.rng.uniform(-0.05, 0.05)
            scores.append((token, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [token for token, _ in scores[:top_k]]

    def update(self, used_tokens: List[str], reward: float, lr: float = 0.1) -> None:
        for token in used_tokens:
            if token not in self.token_weights:
                self.token_weights[token] = 0.0
            self.token_weights[token] += lr * reward


# =========================
# 収束モデル
# =========================

class ConvergenceModel:
    def __init__(self, random_seed: int | None = None) -> None:
        self.rng = random.Random(random_seed)

        # tokenごとの残しやすさ重み
        self.keep_weights: Dict[str, float] = {}

    def select(self, state_tokens: List[str], diverged_tokens: List[str], top_n: int = 3) -> List[str]:
        scored: List[Tuple[str, float]] = []
        state_set = set(state_tokens)

        for token in diverged_tokens:
            score = self.keep_weights.get(token, 0.0)

            # 状態との近さを少し重視
            if token in state_set:
                score += 0.3

            # 短く扱いやすい語を少し優遇
            if len(token) <= 4:
                score += 0.05

            score += self.rng.uniform(-0.03, 0.03)
            scored.append((token, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [token for token, _ in scored[:top_n]]

    def update(self, kept_tokens: List[str], reward: float, lr: float = 0.1) -> None:
        for token in kept_tokens:
            if token not in self.keep_weights:
                self.keep_weights[token] = 0.0
            self.keep_weights[token] += lr * reward


# =========================
# 表層生成
# =========================

def build_response(converged_tokens: List[str]) -> str:
    if not converged_tokens:
        return "うまく言えない。"
    return "".join(converged_tokens) + "。"


# =========================
# ダミー評価器
# 後で Gemini 評価器に差し替える部分
# =========================

class DummyEvaluator:
    def evaluate(self, state_tokens: List[str], candidate: Candidate) -> Evaluation:
        # 仮の評価ロジック
        # 実際には Gemini のスコアに置き換える
        div_score = min(1.0, 0.4 + 0.1 * len(candidate.diverged))
        conv_score = min(1.0, 0.5 + 0.12 * len(candidate.converged))

        naturalness = 0.4
        if candidate.response.endswith("。"):
            naturalness += 0.2
        if 2 <= len(candidate.response) <= 20:
            naturalness += 0.2
        if any(token in candidate.response for token in state_tokens):
            naturalness += 0.1

        efficiency = max(0.0, 1.0 - 0.05 * max(0, len(candidate.diverged) - len(candidate.converged)))

        overall = (
            0.25 * div_score
            + 0.25 * conv_score
            + 0.35 * naturalness
            + 0.15 * efficiency
        )

        return Evaluation(
            divergence_quality=round(div_score, 4),
            convergence_quality=round(conv_score, 4),
            response_naturalness=round(min(naturalness, 1.0), 4),
            thinking_efficiency=round(min(efficiency, 1.0), 4),
            overall_score=round(min(overall, 1.0), 4),
        )


# =========================
# 報酬関数
# =========================

def compute_reward(ev: Evaluation, cost_penalty: float = 0.03) -> float:
    quality = (
        0.25 * ev.divergence_quality
        + 0.25 * ev.convergence_quality
        + 0.35 * ev.response_naturalness
        + 0.15 * ev.thinking_efficiency
    )
    return quality - cost_penalty


# =========================
# 学習ループ本体
# =========================

@dataclass
class LearningStepResult:
    input_tokens: List[str]
    candidates: List[Candidate]
    evaluations: List[Evaluation]
    rewards: List[float]
    best_candidate: Candidate
    best_evaluation: Evaluation
    best_reward: float


class MinimalLSLMLearner:
    def __init__(
        self,
        vocabulary: List[str],
        candidate_count: int = 5,
        divergence_top_k: int = 5,
        convergence_top_n: int = 3,
        learning_rate: float = 0.08,
        random_seed: int | None = None,
    ) -> None:
        self.rng = random.Random(random_seed)
        self.divergence_model = DivergenceModel(vocabulary=vocabulary, random_seed=random_seed)
        self.convergence_model = ConvergenceModel(random_seed=random_seed)
        self.evaluator = DummyEvaluator()

        self.candidate_count = candidate_count
        self.divergence_top_k = divergence_top_k
        self.convergence_top_n = convergence_top_n
        self.learning_rate = learning_rate

    def _make_candidate(self, state_tokens: List[str]) -> Candidate:
        diverged = self.divergence_model.propose(
            state_tokens=state_tokens,
            top_k=self.divergence_top_k,
        )
        converged = self.convergence_model.select(
            state_tokens=state_tokens,
            diverged_tokens=diverged,
            top_n=self.convergence_top_n,
        )
        response = build_response(converged)
        return Candidate(
            diverged=diverged,
            converged=converged,
            response=response,
        )

    def train_step(self, input_tokens: List[str]) -> LearningStepResult:
        candidates: List[Candidate] = []
        evaluations: List[Evaluation] = []
        rewards: List[float] = []

        # 複数候補を作る
        for _ in range(self.candidate_count):
            candidate = self._make_candidate(input_tokens)
            evaluation = self.evaluator.evaluate(input_tokens, candidate)
            reward = compute_reward(evaluation)

            candidates.append(candidate)
            evaluations.append(evaluation)
            rewards.append(reward)

        # 最良候補を選ぶ
        best_index = max(range(len(rewards)), key=lambda i: rewards[i])
        best_candidate = candidates[best_index]
        best_evaluation = evaluations[best_index]
        best_reward = rewards[best_index]

        # ここが「学習」本体
        self.divergence_model.update(
            used_tokens=best_candidate.diverged,
            reward=best_reward,
            lr=self.learning_rate,
        )
        self.convergence_model.update(
            kept_tokens=best_candidate.converged,
            reward=best_reward,
            lr=self.learning_rate,
        )

        return LearningStepResult(
            input_tokens=input_tokens,
            candidates=candidates,
            evaluations=evaluations,
            rewards=rewards,
            best_candidate=best_candidate,
            best_evaluation=best_evaluation,
            best_reward=best_reward,
        )

    def train(self, dataset: List[List[str]], epochs: int = 5) -> None:
        for epoch in range(1, epochs + 1):
            print(f"\n===== EPOCH {epoch} =====")
            for step, input_tokens in enumerate(dataset, start=1):
                result = self.train_step(input_tokens)
                print(
                    f"[STEP {step}] "
                    f"input={input_tokens} "
                    f"best_response={result.best_candidate.response} "
                    f"reward={result.best_reward:.4f} "
                    f"overall={result.best_evaluation.overall_score:.4f}"
                )


# =========================
# サンプル実行
# =========================

def main() -> None:
    vocabulary = [
        "私", "今日は", "少し", "疲れた", "眠い", "休みたい", "仕事",
        "終わった", "お腹", "すいた", "ゲーム", "したい", "外",
        "寒い", "明日", "忙しい", "気分", "いい", "だるい", "散歩"
    ]

    dataset = [
        ["今日は", "疲れた"],
        ["眠い"],
        ["仕事", "終わった"],
        ["少し", "休みたい"],
        ["お腹", "すいた"],
    ]

    learner = MinimalLSLMLearner(
        vocabulary=vocabulary,
        candidate_count=6,
        divergence_top_k=5,
        convergence_top_n=3,
        learning_rate=0.08,
        random_seed=42,
    )

    learner.train(dataset=dataset, epochs=10)

    print("\n===== LEARNED DIVERGENCE WEIGHTS =====")
    for token, w in sorted(learner.divergence_model.token_weights.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{token}: {w:.4f}")

    print("\n===== LEARNED CONVERGENCE WEIGHTS =====")
    for token, w in sorted(learner.convergence_model.keep_weights.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{token}: {w:.4f}")


if __name__ == "__main__":
    main()