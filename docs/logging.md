# LSLM v4 ロギング仕様

## 1. この文書の役割

本書は、LSLM v4 のログとトレースの仕様を定義します。  
v4 においてログは、単なるデバッグ補助ではありません。  
ログは **思考過程を観測し、比較し、改善するための中核装置** です。

特に relation 中心設計では、**どの relation をどうたどったか** を残せなければ、発散と収束を追跡できません。

---

## 2. ログの目的

v4 のログは、少なくとも次を満たす必要があります。

1. 実行失敗を追える
2. 中間段階を追える
3. 同条件比較ができる
4. relation path を追える
5. 学習や分析へ再利用できる
6. 低資源環境でも重くなりすぎない

---

## 3. ログの三層構造

v4 ではログを次の 3 層に分けます。

## 3.1 運用ログ

人間が通常運用で読むログです。

用途:

- 起動確認
- 進捗確認
- warning / error 確認
- relation validation warning 確認
- 終了理由確認

推奨ファイル:

- `runtime/logs/latest.log`

推奨レベル:

- `INFO`
- `WARNING`
- `ERROR`
- `CRITICAL`

---

## 3.2 詳細ログ

開発者が原因調査を行うための詳細ログです。

用途:

- 候補生成数の確認
- relation type ごとの探索量確認
- 閾値挙動の確認
- フォールバック理由の確認
- 例外解析

推奨ファイル:

- `runtime/logs/debug.log`

推奨レベル:

- `DEBUG` 以上すべて

---

## 3.3 構造化トレース

1 入力 1 レコードの JSONL です。  
学習・比較・再現実験・可視化に使います。

推奨ファイル:

- `runtime/traces/latest.jsonl`
- `runtime/traces/latest.session.json`

特徴:

- 人間可読より機械再利用を優先
- 中間状態を段階別に保存
- relation path と timing と score を含む
- 固定設定値や startup 情報は session manifest へ逃がし、各 trace 行へ重複記録しない

---

### 3.4 episode record

loop-learning 用の学習記録です。  
trace をそのまま複製する場所ではなく、**学習判断と改善材料を圧縮して残す層**です。

推奨ファイル:

- `runtime/episodes/latest.jsonl`

特徴:

- `episode_index` と `turn_id` を持つ
- trace 本体は `trace_ref` で参照する
- decision / reward / learning summary / unknown word enrichment を残す
- divergence 候補列や relation path の完全複製はしない

---

## 4. 時刻と識別子

### 4.1 時刻

- JST（Asia/Tokyo）を標準とする
- 人間可読ログは秒精度
- 必要なら trace ではミリ秒またはナノ秒計測を別保持する

### 4.2 識別子

最低限、次を持つことを推奨します。

- `session_id`
- `turn_id`
- `trace_id`

1 入力 1 trace の原則を守るため、`turn_id` は必須とします。

---

## 5. 人間可読ログ形式

基本形式:

```text
YYYY-MM-DD HH:MM:SS [LEVEL] message
```

例:

```text
2026-04-08 20:35:10 [INFO] application_start
2026-04-08 20:35:10 [INFO] lexicon_loaded entries=30020 relations=182004
2026-04-08 20:35:11 [DEBUG] divergence relation_type=hypernym candidates=14
2026-04-08 20:35:11 [WARNING] relation_dangling target=concept:consume source=concept:eat
```

仕様:

- ロガー名は原則省略してよい
- ANSI 色はコンソール専用
- ファイルへ制御文字を残さない

---

## 6. JSONL トレース必須項目

1 入力ごとに最低限次を持ちます。

```json
{
  "session_id": "20260408_203500",
  "turn_id": "20260408_203511_0001",
  "input": "今日は何をするべき？",
  "input_features": {},
  "plan": {},
  "divergence_candidates": [],
  "explored_relations": [],
  "convergence_candidates": [],
  "accepted_relations": [],
  "filled_slots": {},
  "surface_plan": {},
  "response": "...",
  "scores": {},
  "reward": {},
  "timing": {}
}
```

---

## 7. 段階別に何を残すか

## 7.1 Input Analysis

- raw_input
- normalized_input
- unknown_words
- detected_topics
- detected_constraints
- tone_hints

## 7.2 Plan

- intent
- response_mode
- required_slots
- relation_type_priority
- priority
- fallback_reason

## 7.3 Divergence

- seed nodes
- explored relations
- relation type counts
- candidate ids
- candidate scores
- branch depth
- prune reasons

### explored relations の推奨形式

```json
{
  "from": "concept:eat",
  "type": "hypernym",
  "to": "concept:consume",
  "weight": 0.92,
  "reason": "priority:semantic",
  "depth": 1
}
```

## 7.4 Convergence

- accepted candidates
- rejected candidates
- accepted relations
- rejected relations
- ranking reasons
- redundancy penalties
- contradiction flags

## 7.5 Slot

- selected slot frame
- filled slots
- missing slots
- slot evidence

## 7.6 Surface

- sentence plan
- style choice
- template id
- final text
- postprocess notes

## 7.7 External teacher review candidates

- teacher_requests
- teacher_outputs
- teacher_selection
- teacher_improvement_candidates

重要なのは、teacher_hints をそのまま辞書へ戻さず、review candidate として保存することです。

---

## 8. score / reward / timing の扱い

### 8.1 score

構造評価や採用判定のための数値。  
例:

- plan_fitness
- relation_coverage
- divergence_relevance
- convergence_fitness
- relation_precision
- slot_fitness
- grammar_fitness
- dangling_rate

### 8.2 reward

学習接続に使う保存値。  
`internal` / `external` / `total` を基本とします。

### 8.3 timing

最低限、次を残します。

- total_ms
- input_analysis_ms
- plan_ms
- divergence_ms
- convergence_ms
- slot_ms
- surface_ms
- evaluator_ms

これにより、品質低下と遅延増加を分けて追えます。

---

## 9. ローテート方針

起動時に最新ファイルを退避し、新しい `latest.*` を作る方式を推奨します。

対象例:

- `runtime/logs/latest.log`
- `runtime/logs/debug.log`
- `runtime/traces/latest.jsonl`

リネーム例:

- `runtime/logs/20260408203510.log`
- `runtime/logs/20260408203510_debug.log`
- `runtime/traces/20260408203510.jsonl`

衝突時は `_1`, `_2` を付けて退避します。

---

## 10. warning / exception の扱い

### 10.1 warnings

`warnings.warn()` は logging 側へ集約します。

### 10.2 未捕捉例外

未捕捉例外は runtime log と debug log の両方へ記録し、必要なら trace の `failure` セクションにも要約します。

### 10.3 KeyboardInterrupt

通常の停止として扱い、error 汚染を避けます。

### 10.4 relation validation warning

dangling relation や inverse 規約違反は warning として記録し、strict mode では failure として扱います。

---

## 11. 出力量の制御

v4 は低資源前提なので、何でもフル保存すれば良いわけではありません。  
次の 3 モード程度に分けるのが現実的です。

### 11.1 minimal

- 運用ログのみ濃く残す
- trace は要約版

### 11.2 standard

- 通常の開発運用向け
- 1 入力 1 trace を残す

### 11.3 deep_trace

- 候補や relation 経路まで広く残す
- 学習分析やバグ調査向け

モード選択は設定で切り替え、コードへ焼き込まないようにします。

---

## 12. ログでやってはいけないこと

1. `print()` を散在させる
2. 最終応答だけ残して中間状態を捨てる
3. trace と debug log の役割を混ぜる
4. 辞書データ丸ごとを毎回過剰出力する
5. explored path と accepted path を混ぜる
6. 条件比較に必要な識別子を付けない

---

## 13. 結論

LSLM v4 のログは、
**運用ログ・詳細ログ・構造化トレースの三層で、relation path を含む思考過程を段階別に追えること** が核心です。

この仕組みが先に整っていれば、

- 品質の崩れ
- relation 不足
- 遅延の悪化
- 発散不足
- 収束ミス
- slot 欠落

を後から明確に比較できます。  
v4 においてログは後付けではなく、最初から中核機能です。


## 14. trace mode の責務

- `minimal` は件数中心の要約を残す
- `standard` は比較実験に必要な候補と要約を残す
- `deep_trace` は explored relation や slot evidence の生データを残す

つまり `standard` は **読むための trace**、`deep_trace` は **掘るための trace** として分けます。
