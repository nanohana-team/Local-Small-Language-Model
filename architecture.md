# LSLM v4 Architecture

## 1. この文書の役割

この文書は、LSLM v4 の **全体構造** を定義します。  
`docs/philosophy.md` が「何を目指すか」を定義し、`docs/dictionary_schema.md` が「辞書をどう設計するか」を定義し、`docs/relation_design.md` が「relation をどう定義するか」を定義するのに対し、本書は **それらをリポジトリ構成と処理責務に落とした地図** です。

---

## 2. v4 の要約

LSLM v4 は、relation を中核に持つ辞書ネットワークを基盤にして応答を組み立てる軽量言語システムです。  
目標は、次の 4 つを同時に成立させることにあります。

1. **低資源で動くこと**  
   CPU を含む非力な環境でも成立すること。
2. **内部過程を追跡できること**  
   なぜその応答になったかを段階別に見えること。
3. **段階ごとに改善できること**  
   Plan・Divergence・Convergence・Slot・Surface を個別に検証・修正できること。
4. **relation を品質管理できること**  
   どの接続が使われ、どの接続が欠け、どの接続が危険かを追えること。

---

## 3. 現在の実装資産

現リポジトリに実在している主要資産は次の通りです。

### 3.1 辞書 I/O

- `src/core/io/lsd_lexicon.py`
  - JSON / LSD / LSDX のロード
  - 辞書コンテナ正規化
  - 索引自動生成
  - バイナリ保存
  - Indexed 辞書のメモリマップ読込

### 3.2 辞書変換ツール

- `tools/convert_dict_to_binary.py`
  - 辞書形式変換
  - 再ロード検証
  - サイズ比較

### 3.3 設定

- `settings/LLM_order.yaml`
  - 外部 LLM の利用順序定義

### 3.4 設計文書

- `docs/*.md`
  - v4 の思想・relation 仕様・辞書仕様・実装順序・ログ・報酬設計

つまり現在の v4 は、**辞書基盤と設計仕様が先に立っている世代** です。  
会話エンジン本体を急いで増築するのではなく、最初に土台の責務分離を固定する段階にあります。

---

## 4. システム全体像

### 4.1 論理パイプライン

```text
User Input
   ↓
Input Analysis
   ↓
Plan
   ↓
Divergence
   ↓
Convergence
   ↓
Slot Filling
   ↓
Surface Rendering
   ↓
Response
```

### 4.2 横断サブシステム

上記の主経路を、次の横断系が支えます。

- Dictionary System
- Relation System
- Runtime State
- Logging / Trace
- Scoring / Reward
- Evaluator Adapter
- Config / Policy

---

## 5. 主要責務

### 5.1 Input Analysis

入力から次を抽出します。

- 発話タイプ
- 主題候補
- 制約条件
- 感情ヒント
- 会話継続性
- 未知語

ここでは **まだ答えを作らない** ことが重要です。  
役割は「入力理解のための特徴化」に留めます。

### 5.2 Plan

入力に対して、何を達成する返答にするかを決めます。

例:

- 質問に答える
- 理由を説明する
- 比較する
- 手順を提示する
- 感情に寄り添う
- 不足情報を確認する

Plan は自然文ではなく、**応答意図の構造体** です。

### 5.3 Divergence

辞書ネットワーク上で relation を順方向へたどり、関連候補を広げます。  
ここでは量を出すことが仕事であり、まだ最終候補を決めません。

使う主な情報:

- relation
- relation type priority
- axis 近傍
- grammar 制約
- category / hierarchy
- slot 要求

### 5.4 Convergence

Divergence で広げた候補から、現在の Plan を満たすのに必要な relation と要素だけを残します。

主な判定観点:

- Plan 適合
- relation path の説明可能性
- 入力保持
- 矛盾の少なさ
- 冗長性の低さ
- スロット充足への寄与
- 計算コスト

### 5.5 Slot Filling

「誰が・何を・どうした・なぜ・どこで・いつ」のような意味役割を埋めます。  
ここで応答の意味骨格が確定します。

### 5.6 Surface Rendering

確定済みの意味骨格を、日本語として自然な文に変換します。  
この層は **意味を新規に決定してはいけません**。  
意味決定は Plan / Divergence / Convergence / Slot 側で終わっている必要があります。

---

## 6. relation 中心アーキテクチャ

v4 の最重要点は、辞書を単なる知識倉庫ではなく **relation を持つ意味探索空間** として扱うことです。

### 6.1 辞書に含まれるもの

- concept
- lexical entry
- sense
- surface word
- relation
- slot frame
- grammar rule
- axis
- index

### 6.2 relation system が担うもの

- relation type の規約
- relation direction の規約
- inverse 規約
- relation 索引
- dangling target 検証
- import review からの relation 昇格

### 6.3 辞書に含めないもの

- 今回の入力にだけ依存する一時状態
- 現ターン専用の採用候補・棄却候補
- 今回たどった relation path そのもの
- ログや報酬の実測値
- 実験しきい値や LLM 利用順序

この境界を崩すと、辞書が「知識ネットワーク」ではなく「雑多な状態の墓場」になるため、v4 は破綻します。

---

## 7. データ領域の分離

LSLM v4 では、少なくとも次の 4 領域を分離します。

```text
Dictionary  = 安定知識
Runtime     = 現ターンの状態
Records     = ログ・評価・学習記録
Settings    = しきい値・順序・ポリシー
```

### 7.1 Dictionary

安定して再利用される知識ネットワーク。  
relation 本体はここに属します。

### 7.2 Runtime

入力から応答までの一時状態。  
relation の探索候補や path 候補はここに属します。

### 7.3 Records

実行結果を分析・再学習・比較するための保存領域。  
explored relation path と accepted relation path はここへ残します。

### 7.4 Settings

実験条件やしきい値、外部接続順序などを管理する領域。

---

## 8. リポジトリ構成の見方

### 8.1 現在の構成

```text
.
├─ README.md
├─ architecture.md
├─ docs/
├─ settings/
├─ src/
│  └─ core/
│     └─ io/
└─ tools/
```

### 8.2 役割分担

- `docs/`  
  設計そのものを固定する層。
- `settings/`  
  実験条件と外部依存順序を差し替える層。
- `src/core/io/`  
  辞書の読込・保存・正規化という基盤層。
- `tools/`  
  開発補助・変換・検証を行う運用層。

### 8.3 将来的に増えるべき層

```text
src/
├─ apps/            # CLI / chat / learning entrypoints
├─ core/
│  ├─ io/           # 辞書I/O
│  ├─ relation/     # relation schema / index / validation
│  ├─ planning/     # plan生成
│  ├─ divergence/   # 発散
│  ├─ convergence/  # 収束
│  ├─ slotting/     # slot充填
│  ├─ surface/      # 表層化
│  ├─ scoring/      # 内部評価
│  └─ logging/      # trace / runtime log
└─ llm/             # evaluator / teacher adapter
```

この将来構成は「すぐ全部作る」ためではなく、**責務混線を防ぐための設計上の置き場所** です。

---

## 9. 現行辞書実装と目標辞書設計の関係

現行 `lsd_lexicon.py` が扱う辞書コンテナは、互換性と実用性を優先した **lexical entry 中心形式** です。  
一方、v4 の思想上の中心は **concept + relation 中心の知識ネットワーク** にあります。

この差は矛盾ではなく、段階的移行のための二層構造と考えます。

- **現在**: lexical entry を実装上の基本単位として扱う
- **将来**: concept / relation / sense を意味上の中心に据える
- **橋渡し**: lexical entry から concept を参照し、concept から relation をたどれるようにする

詳しい移行案は `docs/dictionary_schema.md` と `docs/relation_design.md` を参照してください。

---

## 10. ログと報酬の位置づけ

### 10.1 ログ

ログは単なるデバッグ補助ではありません。  
v4 では、ログは **思考過程の観測装置** です。

最低限、以下を段階別に追える必要があります。

- input_features
- plan
- divergence candidates
- explored relations
- convergence candidates
- accepted relations
- filled slots
- surface plan
- response
- score / reward
- timing

### 10.2 報酬

報酬は `internal` / `external` / `total` の 3 系統を基本とします。

- `internal`  
  構造的に正しいか
- `external`  
  人間から見て良い応答か
- `total`  
  学習更新に使う統合値

これにより、構造の壊れと表層品質の崩れを分離して扱えます。

---

## 11. v4 で絶対に崩してはいけない判断

1. 辞書と実行時状態を混ぜない
2. relation を曖昧な関連語タグにしない
3. Plan と Surface を混ぜない
4. 発散と収束を 1 つのブラックボックスにしない
5. ログを後付けにしない
6. 外部 LLM を中核ロジックの代用品にしない
7. setting 値をコードへ焼き込まない
8. concept + relation 中心思想を、実装都合だけで捨てない

---

## 12. 結論

LSLM v4 のアーキテクチャは、
**relation を持つ辞書ネットワークを中核に、意味決定・構造化・表層化・観測・評価を責務分離する構造** です。

現在のリポジトリはそのうち、特に

- 辞書 I/O
- 辞書形式変換
- relation と辞書の設計仕様の固定

を先に整えた状態にあります。  
ここから先は「機能を足す」ことよりも、**責務を壊さずに縦へ通すこと** が最優先になります。
