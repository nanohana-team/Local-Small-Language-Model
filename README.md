# Local Small Language Model v4

LSLM v4 は、巨大な LLM の縮小コピーではなく、**低計算資源でも追跡可能に動く relation-first の軽量言語システム**です。

この世代の中核定義は次の 2 文です。

> **辞書は、ただの知識ではなく、知識のネットワークそのものである。**  
> **relation は語句同士・概念同士の接続規則である。**

そのため v4 では、最終応答だけでなく、

- 入力をどう解釈したか
- どの relation をたどったか
- 何を残して何を捨てたか
- どんな意味骨格から日本語を作ったか

を観測できることを重視します。

## このリポジトリの現在地

この版は、v4 の最小縦スライスを **壊れにくい形へ整理した段階** です。

主に次を含みます。

- relation-first chat 実行
- loop-learning 実行
- trace / episode / learning summary の分離
- LSD / LSDX 辞書 I/O
- 辞書メンテナンス用 CLI
- 設計文書群

## 入口はこの 2 本を使う

### 1. アプリ実行

```bash
python main.py --help
python main.py --mode chat --lexicon libs/dict.lsdx
python main.py --mode chat --lexicon libs/dict.lsdx --text "猫は動物？"
python main.py --mode loop-learning --lexicon libs/dict.lsdx --auto-input --max-episodes 32
```

`main.py` は、`chat` と `loop-learning` の共通入口です。

### 2. 辞書メンテナンス

```bash
python tools/lexicon_cli.py --help
python tools/lexicon_cli.py convert-to-binary input.json --verify
python tools/lexicon_cli.py convert-from-binary libs/dict.lsdx --style lexical
python tools/lexicon_cli.py profile-load libs/dict.lsdx --sample-size 256
```

`tools/lexicon_cli.py` は、辞書変換・逆変換・プロファイルの共通入口です。

## リポジトリ構成

```text
.
├─ README.md
├─ architecture.md
├─ main.py
├─ docs/
│  ├─ docs_guide.md
│  ├─ philosophy.md
│  ├─ relation_design.md
│  ├─ dictionary_schema.md
│  ├─ knowledge_boundaries.md
│  ├─ implementation_plan.md
│  ├─ logging.md
│  ├─ reward_design.md
│  └─ script_inventory.md
├─ libs/
│  └─ dict.lsdx
├─ settings/
│  ├─ LLM_order.yaml
│  ├─ scoring_v1.yaml
│  └─ teacher_profiles.yaml
├─ src/
│  ├─ apps/
│  │  ├─ chat_v1.py
│  │  ├─ cli_common.py
│  │  └─ loop_learning_v1.py
│  ├─ core/
│  └─ llm/
├─ tools/
│  ├─ lexicon_cli.py
│  ├─ bootstrap_japanese_lexicon.py
│  ├─ augment_conversation_lexicon.py
│  ├─ convert_dict_to_binary.py
│  ├─ convert_binary_to_dict.py
│  └─ profile_lexicon_load.py
└─ runtime/
   └─ README.md
```

## trace / episode / runtime の分離

### trace

`runtime/traces/latest.jsonl`

- 1 入力 = 1 trace
- 入力特徴、plan、relation 探索、slot、surface、score、timing を残す
- `standard` は比較用、`deep_trace` は調査用

### episode

`runtime/episodes/latest.jsonl`

- loop-learning 専用
- trace の複製ではなく、学習判断の圧縮レコード
- decision / reward / learning summary / unknown enrichment を残す

### learning summary

`runtime/learning_runs/latest.json`

- 1 回の loop-learning 実行全体の要約

### runtime ディレクトリ

`runtime/` は **生成物置き場** です。過去ログや cache はリポジトリ本体へ含めず、空ディレクトリと説明ファイルだけ残します。

## 設計原則

- 辞書を知識ネットワークとして扱う
- relation を辞書中核の接続規則として扱う
- 発散と収束を relation 操作として扱う
- 意味決定と表層化を分離する
- 実行時状態と永続知識を混ぜない
- ログを後付けではなく中核機能として扱う
- 外部 LLM は補助輪に留める
- 低遅延・低資源・再現性を守る

## まず読む順番

1. `docs/docs_guide.md`
2. `docs/philosophy.md`
3. `docs/relation_design.md`
4. `docs/dictionary_schema.md`
5. `architecture.md`
6. `docs/knowledge_boundaries.md`
7. `docs/logging.md`
8. `docs/reward_design.md`
9. `docs/script_inventory.md`

## 補足

既存の個別ツールスクリプトは互換のため残していますが、日常運用の入口は **`main.py` と `tools/lexicon_cli.py`** に寄せる方針です。
