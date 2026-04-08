# LSLM v4 強化学習・報酬設計

## 0. 目的

本仕様書は、LSLM v4 における報酬設計と学習接続の基本方針を定義する。

v4 では、単に最終出力だけを評価して学習するのではなく、  
**知識ネットワーク上での思考過程を段階別に観測し、それぞれに報酬を割り当てる** ことを重視する。

本仕様の目的は以下の通り。

- LSLM v4 の思想に整合した報酬設計を確立する
- 内部評価と外部評価の責務分離を明確化する
- 学習・ログ・分析で扱うデータ構造を固定する
- 将来的な多軸評価への拡張余地を残す

---

## 1. 前提思想

LSLM v4 は、単一のブラックボックス報酬へ依存する設計ではない。  
辞書知識・実行時状態・保存記録・設定を分離し、さらに推論も

- Plan
- Divergence
- Convergence
- Slot
- Surface
- Scoring

へ分割して扱う。

また、辞書は次のように定義する。

> **辞書は、ただの知識ではなく、知識のネットワークそのものである。**

したがって、学習すべき対象は「最終文章の良し悪し」だけではない。  
知識ネットワーク上で何を広げ、何を捨て、何を採用し、どう表層化したかを含めて扱う必要がある。

---

## 2. 報酬設計の結論

v4 における報酬は、少なくとも次の 3 系統に分離する。

```python
reward = {
    "internal": 0.0,
    "external": 0.0,
    "total": 0.0,
}
```

### 2.1 `reward.internal`
LSLM v4 内部の構造的・規則的評価を表す。

### 2.2 `reward.external`
外部評価器による品質評価を表す。

### 2.3 `reward.total`
学習更新や候補選択に用いる統合報酬を表す。

---

## 3. なぜ 3 分離が必要か

### 3.1 思想整合性

LSLM v4 は責務分離を中核原則とする。  
報酬だけを一括スカラーへ潰すと、その思想と矛盾する。

### 3.2 学習工学上の合理性

- `internal` = 安定だが限定的
- `external` = 表現力は高いが揺れやすい

このため、両者を分けて保持した上で統合する方が合理的である。

### 3.3 デバッグ性

3 分離により、

- 内部構造が壊れているのか
- 外部品質だけが低いのか
- 両方が低いのか

を切り分けられる。

---

## 4. reward.internal の定義

### 4.1 役割

`reward.internal` は、LSLM v4 自身の内部構造がどれだけ健全であるかを評価する報酬である。

これは「思考エンジンとしての正しさ」を担保する基礎報酬であり、外部評価器の有無に関わらず算出される。

### 4.2 初期構成要素

最低限以下を持つ。

- `plan_fitness`
- `divergence_relevance`
- `convergence_fitness`
- `slot_fitness`
- `grammar_fitness`
- `input_retention`
- `policy_fitness`（必要に応じて）

### 4.3 基本式

```python
reward.internal = (
    w_plan        * plan_fitness +
    w_divergence  * divergence_relevance +
    w_convergence * convergence_fitness +
    w_slot        * slot_fitness +
    w_grammar     * grammar_fitness +
    w_input       * input_retention +
    w_policy      * policy_fitness
)
```

### 4.4 正規化

```python
reward.internal = max(0.0, min(1.0, reward.internal))
```

---

## 5. reward.external の定義

### 5.1 役割

`reward.external` は、外部評価器が最終応答を評価した結果を実値として取り込む報酬である。

これは「人間から見た品質」に近い信号を導入するための報酬であり、

- 自然さ
- 有用性
- 一貫性
- 応答としての妥当性

などを外部から与える役割を持つ。

### 5.2 evaluator の前提

利用可能 evaluator は以下を想定する。

- Gemini
- OpenAI
- ローカル evaluator

初期段階では、いずれか 1 系統でよい。  
複数 evaluator の比較やアンサンブルは後段階とする。

### 5.3 入力

初期段階で evaluator へ渡す材料は以下とする。

- `user_input`
- `plan`
- `filled_slots`（必要に応じて要約）
- `final_response`
- 補助メタデータ（任意）

ただし、初期段階では evaluator に過剰な中間状態を渡しすぎない。  
まずは最終応答品質の評価器として導入する。

### 5.4 出力

```python
reward.external: float  # 0.0 ~ 1.0
feedback_text: str
label: str
```

### 5.5 正規化

```python
reward.external = max(0.0, min(1.0, evaluator_score))
```

必要に応じて非線形変換を許可するが、初期値は無効でもよい。

---

## 6. reward.total の定義

### 6.1 役割

`reward.total` は、学習更新や候補選択に用いる統合報酬である。

これは `internal` と `external` の両方を反映する唯一の最終値であり、保存対象でもある。

### 6.2 基本式

```python
reward.total = (
    alpha * reward.internal +
    beta  * reward.external
)
```

制約:

```python
alpha + beta = 1.0
```

### 6.3 推奨初期値

```python
alpha = 0.8
beta = 0.2
```

理由:

- 外部評価器ノイズの影響を抑える
- まず内部構造を安定させる
- external を補助信号として導入する

---

## 7. 将来的な段階別拡張

v4 では、将来的に `reward.internal` をさらに分解して保持できるようにする。

例:

```python
reward = {
    "internal": {
        "plan": 0.0,
        "divergence": 0.0,
        "convergence": 0.0,
        "slot": 0.0,
        "surface": 0.0,
        "grammar": 0.0,
        "input_retention": 0.0,
    },
    "external": 0.0,
    "total": 0.0,
}
```

初期実装では平坦化してもよいが、設計上は段階別保持を前提とする。

---

## 8. ログ保存方針

学習ログには最低限以下を保存する。

- 入力
- plan
- divergence 候補数 / 品質
- convergence 採用結果
- filled slots
- final response
- reward.internal
- reward.external
- reward.total
- evaluator feedback
- 所要時間
- 使用辞書版
- 使用設定ハッシュ

これにより、失敗原因の観測可能性を維持する。

---

## 9. 外部LLMとの関係

外部LLMは、あくまで補助用途に限定する。

### 主用途

- evaluator
- 教師データ生成
- 失敗例分析
- 辞書拡張候補の提案

### 禁止事項

- 中核推論を外部LLMへ丸投げしない
- reward.internal の算出を全面依存しない
- 辞書ネットワーク上の思考を代替させない

使用順序や優先度は `settings/LLM_order.yaml` で管理する。

---

## 10. 辞書ネットワークと学習の関係

LSLM v4 において、学習は単に出力文を良くするためだけのものではない。  
学習は、知識ネットワーク上での探索と選別を改善するために存在する。

したがって、学習結果の還元先は以下を含む。

- relation の重み調整候補
- axis 微調整候補
- slot フレーム補強候補
- フォールバック条件の見直し

ただし、辞書への自動反映は禁止し、必ず検証を挟む。

---

## 11. まとめ

LSLM v4 の報酬設計は、

- `reward.internal`
- `reward.external`
- `reward.total`

の 3 分離を基本とし、将来的には段階別内訳まで保持する。

これは、

- 内部構造の健全性
- 人間視点の品質
- 学習更新に使う統合値

を混同しないためである。

特に重要なのは、

> **知識ネットワーク上での思考過程を評価可能にしたまま学習すること**

であり、ここを失うと LSLM v4 の独自性は消える。
