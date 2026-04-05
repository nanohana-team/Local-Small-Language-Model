# architecture.md

## Local Small Language Model 現行アーキテクチャ

この文書は、添付された現在のローカルリポジトリ `Local-Small-Language-Model.zip` の内容を基準に、実際のディレクトリ構造と責務に合わせて整理した現行アーキテクチャ文書である。

---

## 1. 目的

Local Small Language Model（LSLM）は、低計算資源環境で扱える軽量な言語処理基盤を目指すプロジェクトであり、現行リポジトリでは特に以下の基盤部分が実装の中心になっている。

* 語彙辞書の保持形式の標準化
* JSON 辞書から独自バイナリ辞書への変換
* 高速ロード可能な辞書コンテナの実装
* 階層型辞書構造とフラット辞書構造の相互変換
* 将来の発散・収束型推論や学習ループが利用する辞書 I/O 層の整備

現時点の zip に含まれる構成は、以前の構想全体よりもかなり絞られており、**「推論パイプライン全体」よりも「辞書基盤・辞書変換基盤」に重点が置かれた構成**になっている。

---

## 2. 現在のディレクトリ構造

```text
Local-Small-Language-Model/
├─ README.md
├─ README.txt
├─ LICENSE
├─ .env
├─ libs/
│  └─ dict.lsdx
├─ settings/
│  └─ config.yaml
├─ src/
│  ├─ __init__.py
│  └─ core/
│     ├─ __init__.py
│     └─ io/
│        ├─ __init__.py
│        └─ lsd_lexicon.py
└─ tools/
   ├─ convert_dict_to_binary.py
   ├─ dict.json
   └─ dict_v2_2_strict.json
```

補足:

* `.git/` や `__pycache__/` は実行・管理上の補助データであり、アーキテクチャの主対象からは除外する。
* 現行 zip には、以前想定されていた `main.py`、`src/apps/`、`src/llm/`、`src/utils/` などは含まれていない。
* そのため本書では、**現行 zip に実在するファイルのみを前提**に記述する。

---

## 3. 全体アーキテクチャ

現行構成は、大きく次の 4 層で整理できる。

```text
[辞書データ]
    ↓
[辞書I/O層: src/core/io/lsd_lexicon.py]
    ↓
[変換ツール層: tools/convert_dict_to_binary.py]
    ↓
[利用側システム: 将来の推論・学習・評価系]
```

### 3.1 辞書データ層

辞書データ層は、語彙情報そのものを保持する層である。

主なファイル:

* `tools/dict.json`
* `tools/dict_v2_2_strict.json`
* `libs/dict.lsdx`

ここでは以下の情報が辞書エントリとして扱われる。

* 単語
* カテゴリ
* 意味ベクトル (`vector`)
* 文法属性 (`grammar`)
* 階層情報 (`hierarchy`)
* メタ情報 (`meta`)
* 補助インデックス (`indexes`)

### 3.2 辞書 I/O 層

`src/core/io/lsd_lexicon.py` は現行リポジトリの中核であり、辞書データのロード・保存・正規化・変換を一手に担う。

この層の責務は以下の通り。

* JSON 形式辞書の読込
* 独自バイナリ形式 `.lsd` の読込・保存
* インデックス付き独自バイナリ形式 `.lsdx` の読込・保存
* 階層型辞書のフラット化
* フラット辞書から階層コンテナの再構築
* 品詞・開始可能語・終了可能語などの補助インデックス生成
* semantic axis などの meta 補完
* 大規模辞書ロード時の進捗表示
* `.lsdx` の mmap ベース高速アクセス

### 3.3 変換ツール層

`tools/convert_dict_to_binary.py` は、辞書 I/O 層を利用する CLI ツールであり、辞書形式変換をユーザー操作可能にする層である。

主な用途:

* `dict.json` → `dict.lsd` / `dict.lsdx`
* `dict.lsdx` → `dict.json`
* 変換後の件数検証
* 入出力サイズ比較

### 3.4 利用側システム層

現行 zip には未同梱だが、この基盤は本来以下のような上位層から利用される想定である。

* 発散モデル
* 収束モデル
* 学習ループ
* 評価器
* チャット応答系
* 未知語展開器

つまり現行実装は、**将来の探索型言語処理系を支えるための語彙基盤モジュール**として整理するのが正確である。

---

## 4. モジュール責務

## 4.1 `README.md`

プロジェクト全体の思想と目的を記述する文書である。

主な内容:

* LSLM の基本理念
* 発散・収束型思考モデルの考え方
* CPU 前提の軽量動作方針
* 外部 LLM 評価による学習構想
* 従来 LLM と異なる設計思想

役割としては「実装仕様」ではなく、**コンセプト文書兼プロジェクト概要**に近い。

## 4.2 `settings/config.yaml`

モデル選択順序やフォールバック順序を定義する設定ファイルである。

現行内容では `llm-api-order` が定義されており、以下のような順序制御を担う。

* Gemini 系モデルの優先順位
* OpenAI 系モデルのフォールバック
* ローカルモデル指定

ただし、現行 zip にはこの設定を直接読む推論実装は含まれていないため、**上位アプリケーションのための共有設定ファイル**として位置づけるのが適切である。

## 4.3 `src/core/io/lsd_lexicon.py`

現行リポジトリの最重要モジュールである。

### 主責務

* JSON / LSD / LSDLX の統一ロード API 提供
* 階層辞書とフラット辞書の橋渡し
* メタデータとインデックスの自動補完
* バイナリ圧縮・復元
* インデックス付きバイナリ辞書へのランダムアクセス
* 大規模辞書用進捗表示

### 主要構成要素

#### `ConsoleProgressBar`

* コンソール進捗表示
* ロード・保存双方で利用
* 大きい辞書ファイルを扱う際の可視化を担当

#### 基本シリアライズ関数群

* `write_uvarint` / `read_uvarint`
* `write_bytes_with_len` / `read_bytes_with_len`
* `write_str` / `read_str`
* `pack_i16_list` / `unpack_i16_list`
* `quantize_unit_float_to_i16` / `dequantize_i16_to_unit_float`

これらは独自辞書フォーマットの低レベル I/O を担当する。

#### 階層構造操作

* `flatten_hierarchical_lexicon`
* `_flatten_hierarchy_node`
* `build_hierarchical_container_from_entries`

辞書を「編集しやすい階層型 JSON」と「高速処理しやすいフラット辞書」の間で相互変換する。

#### 補完処理

* `_ensure_indexes`
* `_ensure_meta`

辞書エントリから以下を自動生成する。

* `indexes.by_pos`
* `indexes.can_start`
* `indexes.can_end`
* `indexes.content_word`
* `indexes.function_word`
* `indexes.entry_path`
* `meta.semantic_axes`
* `meta.entry_count`

#### ローダー群

* `load_json_lexicon_container`
* `load_lsd_lexicon_container`
* `load_indexed_lsd_lexicon_container`
* `load_lexicon_container`
* `load_lexicon_entries`
* `open_indexed_lexicon`

辞書形式ごとの差異を吸収し、上位層には統一的な辞書コンテナを返す。

#### セーバー群

* `save_json_lexicon_container`
* `save_lsd_lexicon_container`
* `save_indexed_lsd_lexicon_container`
* `save_lexicon_container`

用途に応じて `.json` / `.lsd` / `.lsdx` を保存する。

#### `IndexedLSDLexicon`

* `.lsdx` を `mmap` で直接参照
* 全件展開せずに key 単位で辞書エントリへアクセス
* 大規模語彙辞書を低コストで扱うための高速アクセス機構

これは現行アーキテクチャの中でも、将来的な大語彙対応の要となる設計である。

## 4.4 `tools/convert_dict_to_binary.py`

辞書変換用 CLI エントリポイントである。

### 責務

* 入力ファイル形式の自動判定
* 出力形式の選択
* 再ロード検証
* サイズ比較表示

### 入出力例

```bash
python -m tools.convert_dict_to_binary tools/dict_v2_2_strict.json -o libs/dict.lsdx --format lsdx --verify
```

このツール自体は薄いラッパーであり、実際の変換処理は `src/core/io/lsd_lexicon.py` に委譲される。

---

## 5. データモデル

現行実装では、辞書コンテナはおおむね次のような構造を持つ。

```json
{
  "meta": {
    "semantic_axes": ["..."],
    "entry_count": 0
  },
  "lexicon": {
    "...": {}
  },
  "indexes": {
    "by_pos": {},
    "can_start": [],
    "can_end": [],
    "content_word": [],
    "function_word": [],
    "entry_path": {}
  },
  "entries": {
    "単語": {
      "word": "単語",
      "category": "...",
      "vector": {
        "axis": 0.0
      },
      "grammar": {
        "pos": "..."
      }
    }
  }
}
```

### 5.1 `meta`

辞書全体に関するメタデータを持つ。

主な項目:

* `semantic_axes`
* `grammar_axes`
* `entry_count`

### 5.2 `entries`

単語をキーとするフラット辞書であり、実処理の主対象になる。

各エントリは以下を持つ。

* `word`
* `category`
* `vector`
* `grammar`
* 任意の追加情報

### 5.3 `lexicon`

人間が扱いやすい階層型表現であり、語彙をカテゴリー別に配置する。

現行実装では主に以下の大分類が定義されている。

* `function_words`

  * particles
  * auxiliaries
  * copulas
  * special_marks
* `content_words`

  * pronouns
  * nouns
  * verbs
  * adjectives
  * adverbs
  * conjunctions
  * interjections
  * adnominals
  * prefixes
  * suffixes

### 5.4 `indexes`

高速検索や文法制約判定のための補助インデックス群である。

特に重要なのは以下。

* 品詞別インデックス
* 文頭許可語一覧
* 文末許可語一覧
* 内容語 / 機能語一覧
* 階層パス逆引き

---

## 6. サポートする辞書形式

## 6.1 JSON 形式

用途:

* 人手編集
* デバッグ
* 生成結果の確認
* 設計変更時の可読性確保

長所:

* 編集しやすい
* diff を追いやすい

短所:

* サイズが大きい
* ロードが遅い

## 6.2 `.lsd` 形式

用途:

* 圧縮された独自バイナリ辞書

特徴:

* zlib 圧縮
* semantic vector を int16 に量子化
* JSON より軽量
* 全展開ロード型

## 6.3 `.lsdx` 形式

用途:

* 高速アクセス可能なインデックス付き独自バイナリ辞書

特徴:

* 文字列テーブルを保持
* エントリ位置インデックスを保持
* mmap によるランダムアクセス可能
* 大規模辞書向き

現行実装上、**本命フォーマットは `.lsdx`** と考えてよい。

---

## 7. データフロー

## 7.1 辞書変換フロー

```text
dict.json / dict_v2_2_strict.json
        ↓ load_lexicon_container
内部統一コンテナ化
        ↓ save_lexicon_container
.lsd / .lsdx / .json 出力
```

## 7.2 辞書ロードフロー

```text
入力パス
   ↓
拡張子判定 (.json / .lsd / .lsdx)
   ↓
形式別ローダー実行
   ↓
entries / meta / indexes を含む統一コンテナへ正規化
   ↓
利用側へ返却
```

## 7.3 `.lsdx` 直接アクセスフロー

```text
.lsdx ファイル
   ↓
IndexedLSDLexicon
   ↓
mmap によりインデックス領域を参照
   ↓
必要な key のみデコード
   ↓
エントリ取得
```

---

## 8. 品詞・階層の設計方針

`build_hierarchical_container_from_entries()` では、エントリの `hierarchy` が明示されていない場合でも、`grammar.pos` から既定の格納先を推定して階層へ再配置する。

例:

* `pronoun` → `content_words/pronouns/...`
* `particle_case` → `function_words/particles/case`
* `auxiliary` → `function_words/auxiliaries`
* `copula` → `function_words/copulas`
* `verb` / `verb_stem` → `content_words/verbs/stems/...`
* `adjective_i` → `content_words/adjectives/i/...`
* `adverb` → `content_words/adverbs/...`

この仕組みにより、フラットな辞書しかなくても、内部で一貫した階層辞書を再構築できる。

---

## 9. 設定ファイルの役割

`settings/config.yaml` には、少なくとも現時点で `llm-api-order` が定義されている。

```yaml
llm-api-order:
  - gemini-2.5-flash-lite
  - gemini-2.5-flash
  - gemini-2.0-flash
  - gemini-2.0-flash-lite
  - gpt-5.4-mini
  - gpt-5-mini
  - "local:OpenPipe/Qwen3-14B-Instruct"
```

このファイルの役割は以下の通り。

* 利用可能モデルの優先順位を外出しする
* API 制限時のフォールバック順を制御する
* ローカルモデルと外部モデルを同列に並べる

現行 zip にはこの設定を消費するコードは入っていないが、プロジェクト全体では「辞書基盤」と「モデル呼び出し系」を疎結合に保つための重要な設定点である。

---

## 10. 現在のアーキテクチャ上の特徴

## 10.1 実装の中心が辞書基盤に集約されている

現在の zip では、推論エンジン本体よりも辞書基盤が主実装になっている。

つまり今のアーキテクチャは、

* 思考アルゴリズム本体
* 会話アプリケーション本体
* 学習ループ本体

を直接提供する段階ではなく、**それらを成立させるための辞書インフラを先に固めている段階**と整理できる。

## 10.2 階層表現と高速アクセスの両立を狙っている

JSON の可読性と `.lsdx` の高速性を両方維持する構成になっている。

これは、

* 人間が辞書を設計しやすいこと
* 実行時は大規模辞書を高速に扱えること

の両立を狙った設計である。

## 10.3 追加情報を壊さず保持できる

`grammar` や `entry` に既知項目以外のデータが入っていても、`extras` として保存・復元する設計になっている。

これにより、辞書フォーマットを今後拡張しても互換性を維持しやすい。

---

## 11. 今後この基盤の上に乗る想定の層

現行 zip には未実装または未同梱だが、アーキテクチャ的には以下の層が上に積まれる想定である。

### 11.1 推論層

* 入力トークン列の解析
* 意味近傍探索
* 発散候補生成
* 収束候補選別
* 応答文生成

### 11.2 学習層

* 教師データ生成
* 応答品質評価
* パラメータ更新
* 報酬計算

### 11.3 外部評価層

* Gemini / OpenAI / ローカル LLM による評価
* フォールバック制御
* 評価ログの保存

この意味で、現行 `src/core/io/lsd_lexicon.py` は、将来の上位層から横断的に利用される**基盤中の基盤**である。

---

## 12. 既知の制約

現行 zip の範囲では、以下の制約がある。

* エンドユーザー向けの実行エントリポイントは実質 `tools/convert_dict_to_binary.py` のみ
* 発散・収束アルゴリズム本体は同梱されていない
* 学習ループ、評価器、チャット UI は同梱されていない
* `settings/config.yaml` は定義のみで、同梱コードからは直接参照されていない
* 辞書基盤は強いが、応答生成パイプライン全体はこの zip 単体では完結しない

したがって、このリポジトリを現時点で説明する際は、**「LSLM 全体完成版」ではなく「LSLM の辞書・語彙基盤実装」**として表現するのがもっとも正確である。

---

## 13. まとめ

現行の `Local-Small-Language-Model` ローカルリポジトリは、探索型軽量言語モデルの全機能を一式実装した状態ではなく、主として以下を担う基盤実装で構成されている。

* 階層型語彙辞書の設計
* フラット辞書との相互変換
* 独自バイナリ辞書形式 `.lsd` / `.lsdx` の実装
* 大規模辞書の高速ロードとランダムアクセス
* 将来の推論・学習系が共通利用できる辞書コンテナ API

特に `src/core/io/lsd_lexicon.py` がアーキテクチャ上の中心であり、`tools/convert_dict_to_binary.py` がその利用入口、`tools/*.json` と `libs/dict.lsdx` が実データ層を構成している。

このため、現行アーキテクチャは次のように要約できる。

> **LSLM 現行実装は、発散・収束型言語モデルを成立させるための語彙辞書基盤と、その高速な保存・読込・変換機構を中核とした構成である。**
