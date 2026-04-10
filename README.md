# Local Small Language Model v4

LSLM v4 は、巨大な LLM を小さく複製するプロジェクトではありません。  
**低計算資源で動かせること**、**内部過程を追跡できること**、**辞書を知識ネットワークとして扱えること**を優先した、ホワイトボックス志向の軽量言語システムです。

この世代の中核定義は次の二文です。

> **辞書は、ただの知識ではなく、知識のネットワークそのものである。**  
> **relation は語句同士・概念同士の接続を定義する場である。**

そのため v4 では、単語列をその場で統計的に吐き出すことよりも、

- 入力をどう解釈したか
- どの relation をたどって候補を広げたか
- なぜその relation と候補を採用・棄却したか
- どう意味構造を作り、どう日本語へ表層化したか

を分解・観測・改善できることを重視します。

## 現在のリポジトリの位置づけ

このリポジトリは **v4 の土台を固める初期状態** です。  
現時点で主に入っているのは、辞書 I/O、辞書変換ツール、LLM 使用順序設定、そして設計文書群です。

つまり、いまの v4 は「すでに完成した会話エンジン」ではなく、
**relation 中心の辞書アーキテクチャを破綻なく立ち上げるための最小骨格** として整理されています。

## この版で残している中核資産

- `src/core/io/lsd_lexicon.py`  
  JSON / LSD / LSDX 形式の辞書コンテナを正規化・保存・高速読込する I/O 中核。
- `tools/convert_dict_to_binary.py`  
  辞書を実行用バイナリへ変換する CLI。
- `settings/LLM_order.yaml`  
  外部 LLM 利用時の優先順序設定。
- `docs/*.md`  
  v4 の思想、relation 仕様、辞書仕様、境界、実装順序、ログ、報酬設計を定義する文書群。

## 目指すシステム像

v4 の最小パイプラインは次の通りです。

```text
Input
  ↓
Input Analysis
  ↓
Plan
  ↓
Divergence
  ↓
Convergence
  ↓
Slot
  ↓
Surface
  ↓
Response
```

この流れのうち、いまのリポジトリで最も強いのは **辞書基盤** です。  
今後はこれを軸に、Plan / Divergence / Convergence / Slot / Surface を順に足していきます。

ただし、その中心にあるのは単なる語彙数ではありません。  
v4 が最優先するのは **relation の質** です。  
語が多くても relation が弱ければ辞書は倉庫に留まり、relation が整えば辞書は思考エンジンになります。

## リポジトリ構成

```text
.
├─ README.md
├─ architecture.md
├─ docs/
│  ├─ docs_guide.md
│  ├─ philosophy.md
│  ├─ relation_design.md
│  ├─ dictionary_schema.md
│  ├─ knowledge_boundaries.md
│  ├─ implementation_plan.md
│  ├─ logging.md
│  └─ reward_design.md
├─ settings/
│  └─ LLM_order.yaml
├─ src/
│  ├─ apps/
│  │  └─ chat_v1.py
│  └─ core/
│     ├─ convergence/
│     ├─ divergence/
│     ├─ io/
│     ├─ logging/
│     ├─ planning/
│     ├─ relation/
│     ├─ slotting/
│     └─ surface/
└─ tools/
   └─ convert_dict_to_binary.py
```

## まず読む順番

1. `docs/docs_guide.md`  
   文書全体の役割と読む順番を把握する。
2. `docs/philosophy.md`  
   v4 が何を作るのか、何を作らないのかを確認する。
3. `docs/relation_design.md`  
   relation をなぜ中核に置くのか、発散と収束をどう relation 操作として定義するのかを確認する。
4. `docs/dictionary_schema.md`  
   辞書をどう設計するかを確認する。
5. `architecture.md`  
   リポジトリ構成と処理責務の全体像を確認する。
6. `docs/knowledge_boundaries.md`  
   辞書・実行時・保存・設定の境界を確認する。
7. `docs/implementation_plan.md`  
   実装の優先順序を確認する。
8. `docs/logging.md` / `docs/reward_design.md`  
   観測・評価・学習の規約を確認する。

## すぐに使えるもの

### 最小縦スライス chat v1

relation 基盤 → trace → Plan → Divergence / Convergence → Slot / Surface を最小構成で通す CLI を追加しています。

```bash
python -m src.apps.chat_v1 --lexicon runtime/dictionaries/bootstrapped_v1.json --text "猫は動物？"
```

または `main.py` から次のように呼べます。

```bash
python -m main --mode chat --lexicon runtime/dictionaries/bootstrapped_v1.json --text "猫は動物？"
```

対話モードで起動する場合は `--text` を省略します。
実行すると `runtime/logs/latest.log` と `runtime/traces/latest.jsonl` に記録されます。

### 辞書変換

```bash
python -m tools.convert_dict_to_binary input.json --verify
```

または

```bash
python tools/convert_dict_to_binary.py input.json --verify
```

出力形式は `.json` / `.lsd` / `.lsdx` を扱えます。

## v4 の設計原則

- 辞書を単語表ではなく知識ネットワークとして扱う
- relation を辞書中核の接続規則として扱う
- 発散を relation の順方向探索として扱う
- 収束を relation の逆方向または制約方向探索として扱う
- 意味決定と表層化を分離する
- 中間状態を必ず観測可能にする
- 実行時状態と永続知識を混ぜない
- 設定値をコードに埋め込まない
- 外部 LLM は補助輪であって中核ではない
- 低遅延・低資源・再現性を同時に意識する

## 非目標

- 巨大 LLM と同じ方式で性能競争すること
- ブラックボックスな次トークン予測をそのまま縮小再現すること
- 表面的な自然さだけを先に最適化すること
- 何でも辞書に入れて辞書と実行時を混同すること
- relation を曖昧な関連語タグとして放置すること

## 補足

現行コードの辞書コンテナは **lexical entry 中心の実装互換フォーマット** です。  
ただし v4 の思想上の中心は concept と relation にあります。  
このギャップは `docs/dictionary_schema.md` と `docs/relation_design.md` で、
**現行互換形式から relation 中心設計へどう移るか** まで含めて整理しています。
