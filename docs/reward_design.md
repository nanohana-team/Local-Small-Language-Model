# LSLM v4 報酬設計

## 0. 本書の役割

本書は、LSLM v4 における内部スコア、外部評価、統合報酬の設計を定義する。  
目的は、最終文章だけを雑に採点するのではなく、**知識ネットワーク上での思考過程を段階別に観測し、その上で学習へ接続できる形を作ること** にある。

本書が扱うのは以下である。

- stage scores
- `reward.internal`
- `reward.external`
- `reward.total`
- evaluator 接続方針
- 保存形式

詳細なログ出力先やローテートは `logging.md` に委譲する。

---

## 1. 前提

LSLM v4 は、単一のブラックボックス報酬に依存しない。  
推論は少なくとも次の段階へ分割される。

- Plan
- Divergence
- Convergence
- Slot
- Surface
- Response

また、辞書は知識ネットワークそのものであり、学習対象は「最終文の見た目」だけではない。  
何を広げ、何を捨て、何を採用し、どう文にしたかを含めて評価する必要がある。

---

## 2. 評価の 3 層構造

v4 の評価は、次の 3 層に分ける。

1. **stage scores**
2. **reward.internal**
3. **reward.external / reward.total**

### 2.1 stage scores
各段階の内部的な健全性を個別に評価する値。  
これは最も細かい観測単位であり、デバッグと改善の基礎になる。

### 2.2 reward.internal
stage scores を集約して得る内部報酬。  
外部 evaluator なしでも算出できる。

### 2.3 reward.external / reward.total
外部評価器の信号を加えた報酬。  
学習更新や候補選択に使う最終値は `reward.total` である。

---

## 3. 設計方針の結論

報酬は最低でも次の形で保持する。

```python
reward = {
    "internal": 0.0,
    "external": 0.0,
    "total": 0.0,
}
```

さらに、集約前の情報として stage scores を別に保持する。

```python
scores = {
    "plan_fitness": 0.0,
    "divergence_relevance": 0.0,
    "convergence_fitness": 0.0,
    "slot_fitness": 0.0,
    "surface_fitness": 0.0,
    "grammar_fitness": 0.0,
    "input_retention": 0.0,
}
```

この分離により、

- どの段階が壊れているか
- 内部構造は良いのに外部品質が低いのか
- 表現は良いが構造が壊れているのか

を切り分けられる。

---

## 4. stage scores

### 4.1 目的
stage scores は、各段階を個別に観測・比較・改善するための局所指標である。  
初期段階では、まずここを整える。

### 4.2 初期採用候補

#### `plan_fitness`
入力に対して適切な intent と required slots を立てられているか。

#### `divergence_relevance`
候補群が入力と plan に対して十分に関連しているか。

#### `convergence_fitness`
必要候補を残し、不要候補を落とせているか。

#### `slot_fitness`
必要スロットがどれだけ埋まり、矛盾なく構造化されているか。

#### `surface_fitness`
意味構造を応答として読める形へ落とし込めているか。

#### `grammar_fitness`
文法制約違反が少なく、接続が自然か。

#### `input_retention`
ユーザー入力の主要要素を不当に失っていないか。

### 4.3 性質

- 0.0〜1.0 に正規化する
- evaluator 不要で計算できる形を優先する
- 閾値や重みは設定へ逃がす

---

## 5. reward.internal

### 5.1 役割
`reward.internal` は、LSLM v4 自身の内部構造がどれだけ健全かを評価する集約報酬である。

### 5.2 基本式

```python
reward.internal = (
    w_plan        * scores["plan_fitness"] +
    w_divergence  * scores["divergence_relevance"] +
    w_convergence * scores["convergence_fitness"] +
    w_slot        * scores["slot_fitness"] +
    w_surface     * scores["surface_fitness"] +
    w_grammar     * scores["grammar_fitness"] +
    w_input       * scores["input_retention"]
)
```

### 5.3 正規化

```python
reward.internal = max(0.0, min(1.0, reward.internal))
```

### 5.4 意味
`reward.internal` は、v4 が「思考エンジンとして正しく動いたか」を測る。  
これは外部品質より優先して整えるべき基礎信号である。

---

## 6. reward.external

### 6.1 役割
`reward.external` は、外部 evaluator による品質評価を数値化した報酬である。  
自然さ、有用性、一貫性、妥当性など、人間視点に近い信号を補うために使う。

### 6.2 evaluator の位置づけ
外部 evaluator は補助であり、v4 本体の代替ではない。  
外部評価が高くても内部構造が壊れていれば、それは本質的改善ではない。

### 6.3 初期入力
初期段階では、evaluator へ渡す材料を絞る。

- `user_input`
- `plan` の要約
- `filled_slots` の要約
- `final_response`
- 必要最小限のメタデータ

### 6.4 初期出力

```python
external_feedback = {
    "score": 0.0,
    "label": "",
    "feedback_text": "",
}
```

### 6.5 正規化

```python
reward.external = max(0.0, min(1.0, external_feedback["score"]))
```

---

## 7. reward.total

### 7.1 役割
`reward.total` は、候補選択や学習更新に使う最終統合報酬である。

### 7.2 基本式

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

### 7.3 推奨初期値

```python
alpha = 0.8
beta = 0.2
```

理由:

- 初期段階では内部構造の安定性を重視したい
- external は有益だが揺れやすい
- v4 の思想上、内部可視性を失ってはならない

---

## 8. フェーズ別導入順序

### Phase A. stage scores のみ
まずは内部局所指標だけを出す。  
この段階では学習更新を急がない。

### Phase B. reward.internal 導入
stage scores を集約し、候補比較や実験比較に使える形にする。

### Phase C. reward.external 導入
外部 evaluator を接続し、最終品質の信号を追加する。

### Phase D. reward.total で統合
内部と外部を統合し、学習や候補選択へ利用する。

この順序を逆転させない。  
external 先行は原因分析を難しくする。

---

## 9. 保存形式

各ターンで最低限以下を保存できるようにする。

```json
{
  "turn_id": "20260408_194210_0001",
  "scores": {
    "plan_fitness": 0.0,
    "divergence_relevance": 0.0,
    "convergence_fitness": 0.0,
    "slot_fitness": 0.0,
    "surface_fitness": 0.0,
    "grammar_fitness": 0.0,
    "input_retention": 0.0
  },
  "reward": {
    "internal": 0.0,
    "external": 0.0,
    "total": 0.0
  },
  "external_feedback": {
    "label": "",
    "feedback_text": ""
  }
}
```

`logging.md` で定義するトレースへこの情報を埋め込めるようにする。

---

## 10. 設定へ逃がすべき項目

以下はコード固定ではなく設定で管理する。

- stage score の重み
- `alpha`, `beta`
- clipping の閾値
- evaluator 利用 on/off
- evaluator のモデル順序
- evaluator prompt version

これにより再現実験と比較実験をしやすくする。

---

## 11. やってはいけないこと

### 11.1 最終文だけで採点する
内部崩壊を見逃す。

### 11.2 external を唯一の正解にする
v4 本体の説明可能性が失われる。

### 11.3 stage scores を保存しない
後から何が効いたか検証できなくなる。

### 11.4 一時的な evaluator 反応をそのまま辞書へ入れる
評価結果はまず保存へ置き、知識抽出は別工程で行う。

---

## 12. 結論

LSLM v4 の報酬設計は、「大きな 1 点」で雑に良し悪しを決める設計ではない。  
**段階別に観測し、内部で集約し、必要に応じて外部信号を足す** という三層構造で扱う。

- stage scores で壊れた段階を見つける
- `reward.internal` で内部健全性を集約する
- `reward.external` で人間品質に近い信号を補う
- `reward.total` で最終更新値を作る

この構造を守る限り、v4 の学習は思想と整合したまま前進できる。
