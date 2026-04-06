# LSLM v3 ロギング仕様

## 1. 目的

本仕様は、LSLM v3 で使用する共通ロギング基盤の動作を定義する。

目的は以下の通り。

* `print()` を廃止し、`logging.debug()` / `logging.info()` / `logging.warning()` / `logging.error()` / `logging.critical()` に統一する
* デバッグ情報と通常ログを分離して保存する
* 起動ごとに前回ログを自動ローテートする
* コンソール表示でログレベルを視覚的に判別しやすくする
* 例外や `warnings.warn()` もログへ集約する

---

## 2. ログ出力先

### 2.1 ファイル出力

#### `logs/debug.log`

出力対象:

* `DEBUG`
* `INFO`
* `WARNING`
* `ERROR`
* `CRITICAL`

用途:

* すべての詳細ログを保存する
* 開発時の原因調査に使う

#### `logs/latest.log`

出力対象:

* `INFO`
* `WARNING`
* `ERROR`
* `CRITICAL`

用途:

* 通常運用の進捗と異常を確認する
* `DEBUG` は含めない

### 2.2 コンソール出力

既定では以下を表示する。

* `INFO`
* `WARNING`
* `ERROR`
* `CRITICAL`

`DEBUG` をコンソールに表示したい場合は、`setup_logging(console_level=logging.DEBUG)` を指定する。

---

## 3. ログフォーマット

全出力の基本形式は以下とする。

```text
2026-04-06 12:13:33 [DEBUG] test debug
2026-04-06 12:13:34 [INFO] test info
```

形式:

```text
YYYY-MM-DD HH:MM:SS [LEVEL] message
```

仕様:

* 時刻は JST（Asia/Tokyo）固定
* 秒精度まで出力する
* ロガー名は出力しない
* ファイル出力にもコンソール出力にも同じ文字列形式を使う
* ただしコンソールのみ色付けを行う

---

## 4. コンソール色分け

コンソール表示では、ログレベルごとに以下の色を適用する。

* `DEBUG` : グレー
* `INFO` : 白
* `WARNING` : 黄色
* `ERROR` : 赤
* `CRITICAL` : 赤背景 + 白文字

注意:

* 色付けはコンソール表示のみ
* `logs/debug.log` / `logs/latest.log` には ANSI カラーコードを保存しない
* Windows では ANSI エスケープを有効化して表示する

---

## 5. 起動時ローテート仕様

アプリ起動時、前回実行分の `logs/latest.log` と `logs/debug.log` を自動でリネームする。

### 5.1 リネーム先

* `logs/latest.log` → `logs/YYYYMMDDhhmmss.log`
* `logs/debug.log` → `logs/YYYYMMDDhhmmss_debug.log`

例:

* `logs/latest.log` → `logs/20260406121334.log`
* `logs/debug.log` → `logs/20260406121334_debug.log`

### 5.2 タイムスタンプ決定方法

* まず `latest.log` と `debug.log` の末尾側を読み、最後に記録されたログ行の時刻を抽出する
* その中で新しい方の時刻を採用する
* 時刻が取得できない場合はファイル更新時刻を使う
* 両方とも存在しない場合、または中身が空の場合はローテートしない

### 5.3 衝突時の挙動

同名ファイルが既に存在する場合は、末尾に `_1`, `_2` などを付与して退避する。

---

## 6. 初期化仕様

ロギングはアプリ起動直後に `setup_logging()` を一度だけ呼んで初期化する。

### 6.1 基本使用例

```python
from __future__ import annotations

import logging

from src.utils.logging import setup_logging


def main() -> None:
    setup_logging(app_name="lslm")

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

from src.utils.logging import setup_logging


def main() -> None:
    setup_logging(
        app_name="lslm",
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

* メインスレッドの未捕捉例外
* スレッド内の未捕捉例外

### 7.2 `warnings.warn()`

`logging.captureWarnings(True)` により、Python の warning もログへ流す。

### 7.3 `KeyboardInterrupt`

`KeyboardInterrupt` は通常の終了操作として扱い、標準の挙動を優先する。

---

## 8. 終了時の動作

終了時には `atexit` により終了ログを記録する。

例:

```text
2026-04-06 12:15:00 [INFO] process_exit
```

その後 `logging.shutdown()` によりハンドラを安全に閉じる。

---

## 9. 推奨運用ルール

### 9.1 `print()` の扱い

今後、通常の進捗やデバッグ出力に `print()` は使用しない。

置き換え先:

* `print("...")` → `logging.info("...")`
* 詳細調査用 → `logging.debug("...")`
* 注意喚起 → `logging.warning("...")`
* エラー通知 → `logging.error("...")`
* 致命的エラー → `logging.critical("...")`

### 9.2 ログレベルの使い分け

#### `DEBUG`

* 中間状態
* 分岐判定
* 内部値
* 候補生成の詳細
* デバッグ専用情報

#### `INFO`

* 起動
* 初期化完了
* 通常進捗
* 主要イベント

#### `WARNING`

* 継続可能な異常
* フォールバック発生
* 欠損値補完
* 想定外だが致命的ではない状態

#### `ERROR`

* 処理失敗
* リトライ対象の失敗
* 明確な異常

#### `CRITICAL`

* 継続不能な異常
* 起動失敗
* 主要機能の停止

---

## 10. 提供 API

`src/utils/logging.py` は少なくとも以下を提供する。

### `setup_logging(...)`

ロギング基盤を初期化する。

主な引数:

* `app_name`: アプリ名
* `log_dir`: ログディレクトリ
* `latest_name`: 通常ログファイル名
* `debug_name`: デバッグログファイル名
* `console_level`: コンソール出力の最小ログレベル

### `get_logger(name=None)`

任意名のロガーを取得する。

### `shutdown_logging()`

ロギングを安全に終了する。

### `rotate_previous_logs(...)`

前回ログのローテートを行う。

---

## 11. 想定される実ファイル構成

```text
logs/
├─ latest.log
├─ debug.log
├─ 20260406121334.log
└─ 20260406121334_debug.log
```

---

## 12. 最終要約

本ロギング仕様の要点は以下である。

* すべての通常出力を `logging` に統一する
* `debug.log` に全ログを保存する
* `latest.log` に `INFO` 以上を保存する
* 起動時に前回ログをタイムスタンプ付きで回転する
* コンソールはログレベルごとに色分けする
* 未捕捉例外と warnings もログへ集約する
* 開発時は `console_level=logging.DEBUG` でデバッグ表示を有効にする

この仕様により、開発時の追跡性と運用時の見やすさを両立する。
