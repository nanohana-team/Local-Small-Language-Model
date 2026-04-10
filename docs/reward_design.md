# LSLM v4 報酬・評価設計

## 1. この文書の役割

本書は、LSLM v4 における評価と報酬の設計方針を定義します。  
v4 では、最終出力だけを 1 つの点数で殴る設計を避けます。  
理由は、v4 が **段階別に観測・改善する思想** を持つからです。

relation を中核に置く以上、評価も relation の質を見なければなりません。

---

## 2. 先に結論

v4 の報酬は、少なくとも次の 3 系統に分けます。

```python
reward = {
    "internal": 0.0,
    "external": 0.0,
    "total": 0.0,
}
```

- `internal`  
  内部構造としてどれだけ健全か
- `external`  
  人間から見た応答品質がどれだけ高いか
- `total`  
  学習更新や候補比較に使う統合値

---

## 3. なぜ分けるのか

### 3.1 思想整合性のため

v4 は Plan / Divergence / Convergence / Slot / Surface を分離します。  
それなのに評価だけ 1 つへ潰すと、思想と矛盾します。

### 3.2 デバッグ性のため

- 内部構造は良いが表層が弱い
- 表層は自然だが slot が崩れている
- relation は多いが発散がズレている
- relation は少ないが slot は埋まっている

を分けて扱えるようになります。

### 3.3 学習工学上のため

`internal` は安定しやすく、`external` は表現力が高い代わりに揺れやすいです。  
この性質差を分離して扱う方が合理的です。

---

## 4. internal 評価

## 4.1 役割

`internal` は、LSLM v4 自身の構造的健全性を測る評価です。  
外部 evaluator がなくても算出できる必要があります。

## 4.2 初期評価軸

最低限、次の軸を推奨します。

- `plan_fitness`
- `relation_coverage`
- `divergence_relevance`
- `convergence_fitness`
- `relation_precision`
- `slot_fitness`
- `grammar_fitness`
- `input_retention`
- `latency_fitness`
- `dangling_rate`

### 補足

- `plan_fitness`  
  入力と応答方針が噛み合っているか
- `relation_coverage`  
  必要な接続を辞書が十分持っているか
- `divergence_relevance`  
  発散候補が主題からズレすぎていないか
- `convergence_fitness`  
  必要要素へ正しく絞り込めているか
- `relation_precision`  
  採用した relation が goal にどれだけ寄与しているか
- `slot_fitness`  
  必須 slot が埋まり、矛盾が少ないか
- `grammar_fitness`  
  接続制約や文法面で崩れていないか
- `input_retention`  
  入力の重要点を落としていないか
- `latency_fitness`  
  低遅延制約を大きく破っていないか
- `dangling_rate`  
  参照不能 relation や無効 relation がどれだけ混じっているか

---

## 5. external 評価

## 5.1 役割

`external` は、外部 evaluator が見た最終応答品質です。  
主に次を補います。

- 自然さ
- 有用性
- 一貫性
- 応答としての納得感
- 共感や言い回しの適切さ

## 5.2 evaluator の位置づけ

外部 evaluator は **補助輪** です。  
中核ロジックの代替ではありません。

想定例:

- Gemini
- OpenAI
- ローカル evaluator

## 5.3 入力材料

初期段階では、次を evaluator へ渡せば十分です。

- `user_input`
- `plan` の要約
- `accepted_relations` の要約
- `filled_slots` の要約
- `final_response`

中間状態を全部投げる必要はありません。  
まずは「応答品質の補助評価器」として使います。

---

## 6. total の定義

`total` は、学習更新や候補選択に使う最終値です。

```python
reward.total = alpha * reward.internal + beta * reward.external
```

初期推奨:

```python
alpha = 0.8
beta = 0.2
```

理由:

- まず内部構造を安定させたい
- 外部評価ノイズを過信しない
- 低資源・高再現性の思想に合う

## 6.1 evaluator と teacher を両方使う場合

v4 では external signal を 1 本に潰してから `total` へ入れます。  
ただし、このときも **evaluator と teacher の役割は同一ではない** と考えます。

- `evaluator`  
  最終応答を採点する主 signal
- `teacher`  
  改善案と補助採点を返す副 signal

そのため、両方が利用可能な場合は次のような **主従付き合成** を初期推奨とします。

```python
external = (0.8 * evaluator_score + 0.2 * teacher_score) / (0.8 + 0.2)
```

そして最終報酬は次で計算します。

```python
total = 0.8 * internal + 0.2 * external
```

この形の利点:

- evaluator 不在時でも teacher 単独で external を補完できる
- evaluator があるときは teacher が補助に回るので外部 signal が暴れにくい
- teacher を入れても evaluator の役割が崩れにくい
- external signal の合成規則を trace と設定で追跡できる

重みは固定値でコードへ焼き込まず、設定で差し替えられるようにします。

---

## 7. 保存形式

trace や learning record には、少なくとも次を残します。

```json
{
  "scores": {
    "plan_fitness": 0.82,
    "relation_coverage": 0.79,
    "divergence_relevance": 0.74,
    "convergence_fitness": 0.78,
    "relation_precision": 0.83,
    "slot_fitness": 0.91,
    "grammar_fitness": 0.88,
    "input_retention": 0.84,
    "latency_fitness": 0.95,
    "dangling_rate": 0.00
  },
  "reward": {
    "internal": 0.84,
    "external": 0.76,
    "total": 0.824
  },
  "feedback": {
    "label": "good",
    "text": "結論は明確だが、理由の補足が少ない"
  }
}
```

---

## 8. internal の計算例

```python
reward_internal = (
    0.15 * plan_fitness +
    0.15 * relation_coverage +
    0.10 * divergence_relevance +
    0.15 * convergence_fitness +
    0.10 * relation_precision +
    0.15 * slot_fitness +
    0.08 * grammar_fitness +
    0.07 * input_retention +
    0.03 * latency_fitness +
    0.02 * (1.0 - dangling_rate)
)
```

重みは設定ファイルで差し替えられるようにします。

---

## 9. どの段階を学習対象にするか

v4 では「最終文章だけ」を学習対象にしません。  
段階別に改善対象を持てる設計を維持します。

### 9.1 Plan 改善

- intent のズレ
- required_slots の不足
- relation priority のズレ

### 9.2 Divergence 改善

- 候補不足
- 候補のズレ
- relation 探索の偏り

### 9.3 Convergence 改善

- 重要候補の取りこぼし
- 冗長候補の採用
- 不要 relation の採用

### 9.4 Slot 改善

- 必須 slot の欠落
- 役割の取り違え

### 9.5 Surface 改善

- 文の不自然さ
- 冗長さ
- トーン不一致

---

## 10. 外部 evaluator が使えない場合

外部 evaluator が利用できない場面でも、v4 は止まってはいけません。  
その場合は次の方針を取ります。

- `external = null` または `0.0`
- `total = internal` または internal のみで計算
- fallback reason を trace へ残す

これにより、外部依存の有無でパイプラインが壊れないようにします。

---

## 11. やってはいけないこと

1. 最終応答の印象だけで全てを採点する
2. external score を絶対視する
3. internal score を 1 個だけに潰して詳細を捨てる
4. latency を完全に無視する
5. relation 品質を見ずに語彙数だけで評価する
6. reward の重みをコードへ焼き込む

---

## 12. 結論

LSLM v4 の報酬設計は、
**内部構造の健全性と外部から見た品質を分離し、relation 品質を含めて最後に統合する三層構造** が最適です。

この形なら、

- 学習信号が安定しやすい
- 失敗原因を切り分けやすい
- relation 中心思想と矛盾しない
- 低資源環境でも比較検証しやすい

という v4 の強みを守れます。
