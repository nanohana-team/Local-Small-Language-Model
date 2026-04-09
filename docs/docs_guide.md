# docs ガイド

このディレクトリは、LSLM v4 の設計を **役割ごとに分離して固定するための文書群** です。  
同じ内容を別ファイルに重複して書くのではなく、各ファイルが 1 つの責務を持つように整理しています。

## 読む順番

### 1. `philosophy.md`

最上位思想です。  
「v4 は何を作るのか」「何を作らないのか」「辞書をどう定義するのか」を確認します。

### 2. `relation_design.md`

relation の中心仕様です。  
relation をなぜ辞書中核に置くのか、発散と収束を relation 操作としてどう定義するのか、relation の型・向き・品質管理をどう固定するのかを整理します。

### 3. `dictionary_schema.md`

辞書の中心仕様です。  
concept / lexical entry / sense / relation / slot / grammar / axis を、実装互換性も含めて整理します。

### 4. `knowledge_boundaries.md`

辞書・実行時・保存・設定の境界を定義します。  
どの情報をどこに置くべきかで迷ったら先にここを見ます。

### 5. `implementation_plan.md`

実装順序の定義です。  
何から着手し、どの状態を完了とみなすかを段階ごとに示します。

### 6. `logging.md`

ロギングとトレースの仕様です。  
中間状態の可視化と再現性の基準を定義します。

### 7. `reward_design.md`

評価・報酬・学習接続の仕様です。  
内部構造評価と外部品質評価をどう分けるかを定義します。

## この docs の前提

この docs 群は、次の前提で統一されています。

- 辞書は知識ネットワークである
- relation はノード間の接続規則である
- 発散は relation の順方向探索である
- 収束は relation の逆方向または制約方向探索である
- 意味決定と表層化は分離する
- 中間状態は追跡可能でなければならない
- 実行時状態と永続知識を混ぜない
- 外部 LLM は補助輪であり中核ではない

## 迷ったときの参照先

- 「何を目指すか」で迷った → `philosophy.md`
- 「relation をどう切るか」で迷った → `relation_design.md`
- 「辞書をどう切るか」で迷った → `dictionary_schema.md`
- 「どこに保存するか」で迷った → `knowledge_boundaries.md`
- 「次に何を作るか」で迷った → `implementation_plan.md`
- 「何を記録するか」で迷った → `logging.md`
- 「何を報酬にするか」で迷った → `reward_design.md`
