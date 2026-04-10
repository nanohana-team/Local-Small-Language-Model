# docs ガイド

このディレクトリは、LSLM v4 の設計を **役割ごとに分離して固定するための文書群** です。

## 読む順番

### 1. `philosophy.md`

最上位思想です。

### 2. `relation_design.md`

relation の中核仕様です。

### 3. `dictionary_schema.md`

辞書スキーマの中心仕様です。

### 4. `knowledge_boundaries.md`

辞書・実行時・保存・設定の境界です。

### 5. `implementation_plan.md`

実装順序です。

### 6. `logging.md`

ログと trace の仕様です。

### 7. `reward_design.md`

評価・報酬・学習接続の仕様です。

### 8. `script_inventory.md`

スクリプトの入口・統合方針・runtime 生成物の扱いを整理した文書です。

## 迷ったときの参照先

- 何を目指すか → `philosophy.md`
- relation をどう切るか → `relation_design.md`
- 辞書をどう切るか → `dictionary_schema.md`
- どこに保存するか → `knowledge_boundaries.md`
- 次に何を作るか → `implementation_plan.md`
- 何を記録するか → `logging.md`
- 何を報酬にするか → `reward_design.md`
- どのスクリプトを入口にするか → `script_inventory.md`

## この docs 群の前提

- 辞書は知識ネットワークである
- relation はノード間の接続規則である
- 発散は relation の順方向探索である
- 収束は relation の逆方向または制約方向探索である
- 意味決定と表層化は分離する
- 中間状態は追跡可能でなければならない
- 実行時状態と永続知識を混ぜない
- 外部 LLM は補助輪であり中核ではない
