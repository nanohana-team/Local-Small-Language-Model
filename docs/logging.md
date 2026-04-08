# LSLM v4 ロギング仕様

## 1. 目的

本仕様は、LSLM v4 で使用する共通ロギング基盤の動作を定義する。

v4 におけるログの目的は、単なるデバッグ補助ではない。  
ログは、**知識ネットワーク上で行われた思考過程を観測し、比較し、改善するための中核機能** である。

目的は以下の通り。

- `print()` を廃止し、`logging` に統一する
- 人間可読ログと構造化ログを分離して保存する
- 起動ごとに前回ログを自動ローテートする
- 例外や `warnings.warn()` をログへ集約する
- Plan / Divergence / Convergence / Slot / Surface の各段階を追跡可能にする

---

## 2. ログ出力先

### 2.1 人間可読ログ

#### `runtime/logs/debug.log`

出力対象:

- `DEBUG`
- `INFO`
- `WARNING`
- `ERROR`
- `CRITICAL`

用途:

- すべての詳細ログを保存する
- 開発時の原因調査に使う

#### `runtime/logs/latest.log`

出力対象:

- `INFO`
- `WARNING`
- `ERROR`
- `CRITICAL`

用途:

- 通常運用の進捗と異常を確認する
- `DEBUG` は含めない

### 2.2 構造化ログ

#### `runtime/traces/latest.jsonl`

出力対象:

- 1ターンごとの完全トレース

用途:

- 中間状態の比較
- 学習入力への再利用
- 評価結果の蓄積
- 再現実験

### 2.3 コンソール出力

既定では以下を表示する。

- `INFO`
- `WARNING`
- `ERROR`
- `CRITICAL`

`DEBUG` をコンソールに表示したい場合は、`setup_logging(console_level=logging.DEBUG)` を指定する。

---

## 3. ログフォーマット

### 3.1 人間可読ログの形式

```text
2026-04-08 19:42:10 [INFO] application_start
2026-04-08 19:42:11 [DEBUG] divergence_candidates=12
```

形式:

```text
YYYY-MM-DD HH:MM:SS [LEVEL] message
```

仕様:

- 時刻は JST（Asia/Tokyo）固定
- 秒精度まで出力する
- ロガー名は出力しない
- ファイル出力にもコンソール出力にも同じ基本形式を使う
- コンソールのみ色付けを行う

### 3.2 JSONL トレース形式

1入力につき1レコードを出力する。

```json
{
  "turn_id": "20260408_194210_0001",
  "input": "今日は何をするべき？",
  "plan": {},
  "divergence": [],
  "convergence": [],
  "slots": {},
  "surface": {},
  "response": "...",
  "reward": {},
  "timing": {}
}
```

---

## 4. コンソール色分け

コンソール表示では、ログレベルごとに以下の色を適用する。

- `DEBUG` : グレー
- `INFO` : 白
- `WARNING` : 黄色
- `ERROR` : 赤
- `CRITICAL` : 赤背景 + 白文字

注意:

- 色付けはコンソール表示のみ
- ファイルには ANSI カラーコードを保存しない
- Windows では ANSI エスケープを有効化して表示する

---

## 5. 起動時ローテート仕様

アプリ起動時、前回実行分のログを自動でリネームする。

### 5.1 リネーム対象

- `runtime/logs/latest.log`
- `runtime/logs/debug.log`
- `runtime/traces/latest.jsonl`

### 5.2 リネーム先

- `runtime/logs/latest.log` → `runtime/logs/YYYYMMDDhhmmss.log`
- `runtime/logs/debug.log` → `runtime/logs/YYYYMMDDhhmmss_debug.log`
- `runtime/traces/latest.jsonl` → `runtime/traces/YYYYMMDDhhmmss.jsonl`

### 5.3 タイムスタンプ決定方法

- ログ末尾側から最後の時刻を抽出する
- 抽出に失敗した場合はファイル更新時刻を使う
- 対象が存在しない場合はローテートしない

### 5.4 衝突時の挙動

同名ファイルが既に存在する場合は、末尾に `_1`, `_2` などを付与して退避する。

---

## 6. 初期化仕様

ロギングはアプリ起動直後に `setup_logging()` を一度だけ呼んで初期化する。

### 6.1 基本使用例

```python
from __future__ import annotations

import logging

from src.core.logging.setup import setup_logging


def main() -> None:
    setup_logging(app_name="lslm_v4")

    logging.info("application_start")
    logging.debug("debug message")
    logging.warning("warning message")
    logging.error("error message")


if __name__ == "__main__":
    main()
```

### 6.2 デバッグをコンソールに出す例

```python
from __future__ import annotations

import logging

from src.core.logging.setup import setup_logging


def main() -> None:
    setup_logging(
        app_name="lslm_v4",
        console_level=logging.DEBUG,
    )

    logging.debug("test debug")
    logging.info("test info")
    logging.warning("test warning")
    logging.error("test error")
    logging.critical("test critical")


if __name__ == "__main__":
    main()
```

---

## 7. 例外と warnings の扱い

### 7.1 未捕捉例外

未捕捉例外はロギング基盤がフックし、ログへ記録する。

対象:

- メインスレッドの未捕捉例外
- スレッド内の未捕捉例外

### 7.2 `warnings.warn()`

`logging.captureWarnings(True)` により、Python の warning もログへ流す。

### 7.3 `KeyboardInterrupt`

`KeyboardInterrupt` は通常の終了操作として扱い、標準の挙動を優先する。

---

## 8. JSONL トレースの必須項目

LSLM v4 の JSONL トレースには最低限以下を含める。

- `turn_id`
- `input`
- `input_features`
- `plan`
- `divergence_candidates`
- `convergence_candidates`
- `filled_slots`
- `surface_plan`
- `response`
- `reward`
- `timing`
- `fallbacks`
- `dict_version`
- `config_hash`

特に `divergence_candidates` と `convergence_candidates` は、知識ネットワーク上でどのノード・関係をたどったかが分かる形で保持する。

---

## 9. 推奨運用ルール

### 9.1 `print()` の扱い

通常の進捗やデバッグ出力に `print()` は使用しない。

### 9.2 ログレベルの使い分け

#### `DEBUG`
- 中間状態
- 分岐判定
- 内部値
- 候補生成詳細
- relation 探索経路

#### `INFO`
- 起動
- 初期化完了
- 通常進捗
- 主要イベント

#### `WARNING`
- 継続可能な異常
- フォールバック発生
- 欠損補完
- 想定外だが致命的ではない状態

#### `ERROR`
- 処理失敗
- 入出力エラー
- 候補生成不能

#### `CRITICAL`
- 実行継続が困難な致命的障害

---

## 10. 設計原則

### 10.1 ログは補助ではなく中核機能
ログは後から眺めるだけのものではなく、改善判断を支える基盤である。

### 10.2 1ターン完全追跡
1入力に対し、1トレースで中間状態まで再現できること。

### 10.3 比較可能性
異なる辞書版、設定版、実装版の差分を比較できること。

### 10.4 学習再利用性
トレースがそのまま学習や分析へ再利用できること。

---

## 11. まとめ

LSLM v4 のログは、

- 人間が読むためのログ
- 機械が再利用するためのトレース

の二層で構成する。  
特に JSONL トレースは、知識ネットワーク上で何を広げ、何を捨て、何を採用したかを残すための中核記録である。
