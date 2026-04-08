# LSLM v4 ロギング仕様

## 0. 本書の役割

本書は、LSLM v4 の観測基盤を定義する。  
目的は、デバッグのためだけではなく、**思考過程を比較・分析・改善するための共通ログ基盤** を固定することである。

本書が扱うのは以下である。

- ログの種類
- 出力先
- レベル運用
- トレース構造
- ローテート
- 例外・警告の集約

思想は `philosophy.md`、実装順序は `implementation_plan.md`、報酬設計は `reward_design.md` に委譲する。

---

## 1. ログの位置づけ

LSLM v4 におけるログは、単なる `print()` の置き換えではない。  
ログは、**知識ネットワーク上で何を参照し、何を広げ、何を捨て、何を採用したか** を観測するための中核機能である。

そのため v4 では、以下の 3 層を分けて扱う。

1. 人間可読ログ
2. 構造化トレース
3. 評価・学習向け記録

本書は 1 と 2 を主対象とし、3 との接続点も最低限定義する。

---

## 2. 出力先

### 2.1 人間可読ログ

#### `runtime/logs/latest.log`
対象:

- `INFO`
- `WARNING`
- `ERROR`
- `CRITICAL`

用途:

- 日常運用の確認
- 異常検知
- 実行の大まかな流れの把握

#### `runtime/logs/debug.log`
対象:

- `DEBUG`
- `INFO`
- `WARNING`
- `ERROR`
- `CRITICAL`

用途:

- 詳細調査
- 候補数や棄却理由の確認
- 開発時の深掘り

### 2.2 構造化トレース

#### `runtime/traces/latest.jsonl`
対象:

- 1 ターンごとの完全トレース

用途:

- 中間状態の比較
- 再現実験
- 学習入力への再利用
- evaluator への再投入

### 2.3 コンソール

既定では以下を表示する。

- `INFO`
- `WARNING`
- `ERROR`
- `CRITICAL`

必要に応じて `console_level=logging.DEBUG` を許可する。

---

## 3. レベル運用

### 3.1 `DEBUG`

- 候補数
- relation 展開件数
- pruning 詳細
- grammar check 詳細
- 各 stage の中間状態

### 3.2 `INFO`

- 起動 / 終了
- 辞書ロード成功
- 1 ターン開始 / 終了
- evaluator 接続成功
- モード変更

### 3.3 `WARNING`

- フォールバック発生
- 候補不足
- 想定外だが継続可能な状態
- evaluator タイムアウト時の代替動作

### 3.4 `ERROR`

- 該当ターンの失敗
- 辞書読み込み失敗
- 必須構造不足
- JSONL 出力失敗

### 3.5 `CRITICAL`

- 実行継続不能
- ログ初期化自体の失敗
- コアデータ破損

---

## 4. 人間可読ログ形式

形式は以下とする。

```text
YYYY-MM-DD HH:MM:SS [LEVEL] message
```

例:

```text
2026-04-08 19:42:10 [INFO] application_start
2026-04-08 19:42:11 [DEBUG] divergence_candidates=12
```

仕様:

- 時刻は JST（Asia/Tokyo）固定
- 秒精度まで出力する
- ロガー名は原則出さない
- ファイルとコンソールで基本形式を揃える
- 色付けはコンソールのみに適用する

---

## 5. トレースの基本方針

### 5.1 1 ターン 1 レコード
1 つの user input に対して 1 つの JSONL レコードを出力する。

### 5.2 ステージごとに構造を分ける
Plan / Divergence / Convergence / Slot / Surface / Response を別キーとして保持する。

### 5.3 棄却理由を残す
採用結果だけでなく、何をなぜ捨てたかを残す。

### 5.4 timing を必ず残す
全体時間だけでなく、可能なら stage 単位の所要時間も残す。

---

## 6. JSONL トレースの最小スキーマ

```json
{
  "turn_id": "20260408_194210_0001",
  "timestamp_jst": "2026-04-08T19:42:10+09:00",
  "input": {
    "raw": "今日は何をするべき？",
    "features": {}
  },
  "plan": {},
  "divergence": {
    "seeds": [],
    "candidates": [],
    "paths": []
  },
  "convergence": {
    "selected": [],
    "rejected": []
  },
  "slots": {
    "required": [],
    "filled": {},
    "missing": []
  },
  "surface": {
    "plan": {},
    "drafts": []
  },
  "response": {
    "text": "",
    "status": "ok"
  },
  "scores": {},
  "reward": {},
  "timing": {}
}
```

`reward` は未導入段階では空でもよい。  
ただしキー自体は将来互換性のため確保してよい。

---

## 7. ステージ別の記録項目

### 7.1 Input / Feature

- raw input
- normalization result
- extracted keywords
- detected mode
- conversation metadata（必要な範囲のみ）

### 7.2 Plan

- intent
- response_mode
- required_slots
- constraints
- fallback_policy

### 7.3 Divergence

- seeds
- relation expansion paths
- axis-neighbor hits
- candidate pool
- candidate count

### 7.4 Convergence

- selected candidates
- rejected candidates
- rejection reasons
- score breakdown

### 7.5 Slot

- required slots
- filled slots
- missing slots
- slot confidence

### 7.6 Surface

- surface plan
- phrasing candidates
- grammar violations
- selected phrasing reason

### 7.7 Response

- final text
- completion status
- fallback used or not

---

## 8. 例外と警告の集約

### 8.1 warnings
`warnings.warn()` は logging へ集約する。

### 8.2 uncaught exceptions
未捕捉例外は必ず `ERROR` 以上で記録する。

### 8.3 stage context
例外時は、可能な限り以下を付与する。

- turn_id
- stage_name
- active input summary
- current candidate counts

これにより「どの段階で落ちたか」を即座に追えるようにする。

---

## 9. 起動時ローテート

アプリ起動時、前回実行分のログを自動退避する。

### 9.1 対象

- `runtime/logs/latest.log`
- `runtime/logs/debug.log`
- `runtime/traces/latest.jsonl`

### 9.2 退避先

- `runtime/logs/YYYYMMDDhhmmss.log`
- `runtime/logs/YYYYMMDDhhmmss_debug.log`
- `runtime/traces/YYYYMMDDhhmmss.jsonl`

### 9.3 タイムスタンプ決定

優先順位は次の通り。

1. ログ末尾から最後の時刻を抽出
2. 抽出失敗時はファイル更新時刻を使用
3. 対象不存在時は何もしない

### 9.4 衝突時
同名が存在する場合は `_1`, `_2` を付与して退避する。

---

## 10. 初期化仕様

ロギングはアプリ起動直後に `setup_logging()` を一度だけ呼んで初期化する。

要件:

- 複数回初期化で handler 重複を起こさない
- ログディレクトリを自動作成する
- コンソール / latest / debug の 3 出力を同時に扱える
- trace writer を別管理できる

---

## 11. 実務ルール

### 11.1 `print()` を常用しない
一時デバッグ以外は logging へ統一する。

### 11.2 候補数だけでなく理由を残す
「候補 0 件」だけでは弱い。  
なぜ 0 件になったかを残す。

### 11.3 長文データはログとトレースで役割を分ける
人間可読ログへ巨大 JSON を垂れ流さない。  
詳細構造は JSONL トレースへ置く。

### 11.4 失敗時ほど情報を多く残す
正常時よりも異常時のほうが分析価値が高い。  
例外経路には stage 情報を積極的に付与する。

---

## 12. reward 設計との接続

`reward_design.md` で定義する以下の情報は、トレースに保存できるようにしておく。

- stage scores
- `reward.internal`
- `reward.external`
- `reward.total`
- evaluator feedback summary

ただし、報酬計算ロジックそのものは本書の責務ではない。

---

## 13. 結論

LSLM v4 のログは、単に「何か起きた」を記録するものではない。  
**思考過程を後から再構成できること** が価値である。

- 人間可読ログで状況を追う
- JSONL トレースで内部状態を追う
- 例外・警告を集約して壊れ方を追う

この 3 層が成立して初めて、v4 は改善可能なシステムになる。
