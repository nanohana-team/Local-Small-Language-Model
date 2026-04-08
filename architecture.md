# LSLM v4 Architecture

## 1. 概要

LSLM v4 は、一般的な大規模言語モデル（LLM）の縮小版を目指すものではない。
本プロジェクトは、**低計算資源環境でも動作可能**であり、かつ**内部過程を追跡・分解・改善できる**会話AIの構築を目的とする。

v4 における中核思想は以下の 3 点である。

1. **意味決定と表層生成の分離**
2. **辞書を「操作可能な意味空間」として扱うこと**
3. **各処理段階を個別に観測・評価・学習可能にすること**

これにより、LSLM v4 はブラックボックスな次トークン予測器ではなく、
**計画（Plan）→意味充填（Slot）→表層生成（Surface）** の責務分離を持つ、
説明可能な小型言語処理系として設計される。

---

## 2. 設計目標

### 2.1 主目標

* CPU を含む低計算資源環境で動作可能であること
* 応答生成の各段階を追跡可能であること
* 各段階を個別に改善可能であること
* 辞書を基盤とした発散・収束処理を中核に据えること
* 外部LLMに依存しすぎず、内部推論を自立化すること

### 2.2 副目標

* 学習データ・ログ・評価結果を構造化して保存すること
* モジュール単位で差し替え可能な設計にすること
* 将来的な並列学習・オンライン改善に耐えること

### 2.3 非目標

* 汎用LLMと同等の知識量を単体で保持すること
* 巨大パラメータモデルと同じ方式で競争すること
* 表層的な自然さのみを最適化すること
* 内部過程を犠牲にして応答品質だけを追うこと

---

## 3. v4 における基本方針

### 3.1 責務分離を絶対に崩さない

v4 では、応答生成を以下の 3 層に分離する。

* **Plan Layer**

  * ユーザー入力から、何を答えるべきかの骨子を決定する
* **Slot Layer**

  * Plan を満たすために必要な意味要素・概念・語彙候補を確定する
* **Surface Layer**

  * 確定済みの意味構造を自然な文章として整形する

この分離により、
「何を言うべきかの誤り」と「どう言うかの不自然さ」を分けて扱える。

### 3.2 辞書を意味空間として扱う

辞書は単なる単語一覧ではない。
LSLM v4 における辞書は、以下を保持する意味操作基盤である。

* 語彙エントリ
* 品詞・カテゴリ
* 意味軸ベクトル
* 関連語・近傍語
* 上位概念・下位概念
* 使用傾向・重み
* 発散候補生成用メタ情報
* 収束候補評価用メタ情報

### 3.3 各段階を学習可能にする

v4 は最終出力のみを評価しない。
各段階に対して個別の評価軸と報酬を持つ。

* Plan の妥当性
* Slot の充足度
* Surface の自然さ
* 計算量・遅延・冗長性

これにより、どこで失敗したかを明確にしたまま改善できる。

---

## 4. 全体アーキテクチャ

### 4.1 論理構成

```text
User Input
   ↓
[Input Analysis]
   ↓
[Plan Layer]
   ↓
[Semantic Divergence]
   ↓
[Semantic Convergence]
   ↓
[Slot Layer]
   ↓
[Surface Layer]
   ↓
Response Text
```

この主経路を、以下の支援系が横断的に支える。

* Dictionary / Lexicon System
* Evaluation System
* Learning Loop
* Logging / Trace System
* External LLM Adapter
* Config / Runtime Control

---

## 5. 処理フロー

### 5.1 Input Analysis

入力文を解析し、以下を抽出する。

* 発話種別（質問 / 命令 / 雑談 / 応答継続）
* 主題候補
* 制約条件
* 感情的トーン
* 会話履歴依存の有無
* 不明語・未知語

出力は **入力特徴構造（Input Features）** とする。

---

### 5.2 Plan Layer

Input Features をもとに、応答の骨子を決定する。

Plan は自然文ではなく、意味的な意図構造として表現される。
例:

* 質問に答える
* 理由を説明する
* 比較する
* 手順を提示する
* 感情に寄り添う
* 不足情報を補う

Plan Layer の目的は、**「何を達成する応答か」** を決めることにある。

出力は以下を含む。

* intent
* response_mode
* required_slots
* constraints
* priority

---

### 5.3 Semantic Divergence

Plan を起点に、辞書空間から関連概念を広げる。

この処理は単なる連想ではなく、
**Plan に沿った意味候補の探索** である。

主な役割:

* 関連語候補の抽出
* 概念近傍の探索
* 不足語の補完候補生成
* 複数視点の候補展開
* 曖昧語の意味分岐

発散は無制限に広げない。
探索深度、分岐数、スコア閾値、カテゴリ制約を設ける。

---

### 5.4 Semantic Convergence

発散で得られた候補群から、Plan に必要な意味要素を絞り込む。

収束は、以下の観点で行う。

* Plan との整合性
* 入力との関係性
* 応答目的への貢献度
* 冗長性の低さ
* 矛盾の少なさ
* コスト効率

この段階では、まだ文章を作らない。
**意味候補を選別して、必要な要素だけ残す** のが役割である。

---

### 5.5 Slot Layer

収束済み候補から、応答に必要な意味スロットを埋める。

例:

* 主題
* 比較対象
* 根拠
* 手順
* 注意点
* 結論
* 感情的配慮

Slot Layer は、Plan が要求する構造を満たすことを目的とする。
ここで応答内容の骨格が確定する。

出力は **意味スロット構造（Slot Structure）** であり、
これは自然文生成前の最終意味表現となる。

---

### 5.6 Surface Layer

Slot Structure を自然な文章へ変換する。

この層の責務は以下に限定される。

* 文順の決定
* 接続表現の挿入
* 文体調整
* 冗長表現の削減
* 日本語の自然性向上
* 出力フォーマット整形

Surface Layer は、
**意味を新規に決定してはならない**。
意味決定は Plan / Slot 側で完了している必要がある。

---

## 6. 中核コンポーネント

### 6.1 Dictionary System

辞書システムは v4 の基盤である。

#### 役割

* 語彙エントリ管理
* 意味軸管理
* 関係語管理
* カテゴリ管理
* 発散・収束補助情報保持
* バイナリ高速読込
* JSON / JSONL / 独自形式相互変換

#### 要件

* 高速ロード可能であること
* 部分更新可能であること
* 検証・整合性チェックが可能であること
* ログから辞書更新根拠を追跡可能であること

#### 辞書形式

v4 では、実行時高速アクセス形式と編集用形式を分離する。

* 編集用: JSON / JSONL
* 実行用: lsdx などの独自バイナリ形式
* 補助索引: インデックスファイル

---

### 6.2 Planner

Planner は Input Features を受け取り、Plan を生成する。

#### 役割

* 発話意図判定
* 応答モード決定
* 必要スロット定義
* 優先順位付け
* 失敗時のフォールバック方針決定

#### 出力例

```json
{
  "intent": "explain_reason",
  "response_mode": "structured_explanation",
  "required_slots": ["topic", "reason", "conclusion"],
  "constraints": ["keep_concise", "use_japanese"],
  "priority": 0.82
}
```

---

### 6.3 Diverger

Diverger は辞書空間上で関連候補を拡張する。

#### 役割

* 近傍語探索
* 軸ベース候補拡張
* カテゴリ横断候補生成
* 不明語周辺探索
* 多視点候補抽出

#### 制約

* 無制限再帰禁止
* 深度制限必須
* 分岐数制限必須
* 低スコア枝の早期打ち切り

---

### 6.4 Converger

Converger は発散候補から必要要素のみを選別する。

#### 役割

* 候補スコアリング
* 重複排除
* 矛盾排除
* Plan 適合度評価
* Slot 充足候補の抽出

#### 出力

* 上位候補群
* 削除理由
* 最終採用理由
* スコア内訳

---

### 6.5 Slot Builder

Slot Builder は収束結果を、応答構造として解釈可能な形に整理する。

#### 役割

* スロット定義に基づく意味要素配置
* 不足スロット検知
* フォールバック生成
* 応答骨格組み立て

#### 特徴

* 自然文生成を行わない
* 意味構造のみを出力する
* Surface Layer へ明確な契約を渡す

---

### 6.6 Surface Realizer

Surface Realizer は意味構造を自然文にする。

#### 役割

* 文テンプレート選択
* 文体選択
* 接続・並列・強調の調整
* 文長制御
* 改行・箇条書き整形

#### 制約

* 新しい意味を追加しない
* Slot Structure にない内容を勝手に補わない
* 表層の自然さに専念する

---

### 6.7 External LLM Adapter

外部LLMは補助用途に限定する。

#### 主用途

* 教師データ生成
* 評価補助
* 未知語補完
* 辞書拡張候補生成
* 失敗例の分析補助

#### 禁止事項

* 毎回の中核推論を外部LLMに依存しない
* Plan / Slot 決定を丸投げしない
* LSLM を単なる LLM ラッパーにしない

---

### 6.8 Evaluation System

評価は段階別に行う。

#### 評価対象

* Plan Quality
* Divergence Quality
* Convergence Quality
* Slot Completeness
* Surface Naturalness
* Response Utility
* Compute Cost

#### 方針

最終応答だけでなく、各中間生成物を評価対象に含める。
これにより、失敗箇所を特定可能にする。

---

### 6.9 Learning Loop

学習ループは、各段階の誤差を局所的に改善することを目的とする。

#### 学習対象例

* Planner の意図決定
* Diverger の候補展開傾向
* Converger の選別重み
* Slot Builder の構造化精度
* Surface Realizer の表現選択

#### 学習方針

* 段階別報酬
* 失敗例蓄積
* 再現可能エピソード保存
* コスト制約付き最適化
* 教師あり + 強化学習 + ルール調整の併用

---

## 7. データモデル

### 7.1 Input Features

```json
{
  "text": "...",
  "detected_intents": [],
  "entities": [],
  "constraints": [],
  "emotion_hint": null,
  "unknown_tokens": []
}
```

### 7.2 Plan

```json
{
  "intent": "...",
  "response_mode": "...",
  "required_slots": [],
  "constraints": [],
  "priority": 0.0
}
```

### 7.3 Candidate Node

```json
{
  "token": "...",
  "category": "...",
  "axis_vector": [],
  "score": 0.0,
  "source": "...",
  "relations": []
}
```

### 7.4 Slot Structure

```json
{
  "topic": null,
  "reason": [],
  "examples": [],
  "caution": [],
  "conclusion": null
}
```

### 7.5 Trace Record

```json
{
  "turn_id": "...",
  "input": {},
  "plan": {},
  "divergence": [],
  "convergence": [],
  "slots": {},
  "surface": {},
  "reward": {},
  "timing": {}
}
```

---

## 8. ログ・トレース設計

v4 においてログは補助ではなく中核機能である。
目的は「あとで読む」ことではなく、**改善判断を即時に下せる比較可能性を持たせること** にある。

### 必須ログ項目

* 入力
* Input Features
* Plan
* 発散候補一覧
* 収束採用候補一覧
* スロット確定結果
* 最終出力
* 各段階のスコア
* 各段階の所要時間
* 失敗理由
* 外部LLM利用有無
* 使用辞書バージョン
* 設定ファイルハッシュ

### ログ形式

* 人間可読ログ: `.log`
* 構造化ログ: `.jsonl`
* 集計用ログ: `.json`

### 要件

* ターン単位で完全追跡可能
* 再実行比較可能
* 設定差分比較可能
* 学習入力へ再利用可能

---

## 9. 学習・評価設計

### 9.1 報酬分解

v4 では総合報酬のみを持たない。
最低でも以下に分解する。

* `reward.plan`
* `reward.divergence`
* `reward.convergence`
* `reward.slot`
* `reward.surface`
* `reward.utility`
* `reward.cost`
* `reward.total`

#### 例

```json
{
  "reward": {
    "plan": 0.84,
    "divergence": 0.71,
    "convergence": 0.78,
    "slot": 0.88,
    "surface": 0.69,
    "utility": 0.81,
    "cost": -0.22,
    "total": 0.74
  }
}
```

### 9.2 評価原則

* 各段階の失敗は各段階で測る
* 最終出力だけで内部改善を判断しない
* 低コスト性も性能指標に含める
* 外部LLM評価は補助として扱う
* 評価器の偏りをログに残す

---

## 10. 実行モード

v4 では少なくとも以下の実行モードを想定する。

### chat

対話用。Plan → Slot → Surface を一通り実行する。

### inspect

中間生成物を詳細表示する解析用モード。

### learn

学習ループを回し、評価結果を蓄積する。

### batch-eval

既存データセットに対して一括評価する。

### dict-build

辞書変換・検証・索引生成を行う。

### dict-check

辞書整合性検査を行う。

---

## 11. ディレクトリ構成案

```text
LSLM/
├─ main.py
├─ settings/
│  ├─ config.yaml
│  ├─ planner.yaml
│  ├─ divergence.yaml
│  ├─ convergence.yaml
│  ├─ surface.yaml
│  └─ reward.yaml
├─ libs/
│  ├─ dict.json
│  ├─ dict.lsdx
│  └─ indexes/
├─ src/
│  ├─ apps/
│  │  ├─ run_chat.py
│  │  ├─ run_learn.py
│  │  ├─ run_inspect.py
│  │  └─ run_batch_eval.py
│  ├─ core/
│  │  ├─ input/
│  │  ├─ plan/
│  │  ├─ divergence/
│  │  ├─ convergence/
│  │  ├─ slots/
│  │  ├─ surface/
│  │  ├─ evaluation/
│  │  ├─ learning/
│  │  └─ logging/
│  ├─ dict/
│  │  ├─ loader.py
│  │  ├─ validator.py
│  │  ├─ converter.py
│  │  └─ indexer.py
│  ├─ llm/
│  │  ├─ adapters/
│  │  └─ evaluators/
│  └─ utils/
├─ tools/
│  ├─ build_dict.py
│  ├─ verify_dict.py
│  └─ convert_dict.py
├─ runtime/
│  ├─ logs/
│  ├─ traces/
│  ├─ datasets/
│  ├─ sessions/
│  └─ eval/
└─ README.md
```

---

## 12. 失敗時の設計原則

v4 は失敗を前提に設計する。

### 原則

* 発散失敗時は shallow fallback を行う
* 収束失敗時は Plan に戻る
* スロット不足時は不足明示応答を生成可能にする
* 表層生成失敗時は簡易テンプレートで救済する
* 外部LLM補助発動時は必ずログに残す

---

## 13. 今後の拡張余地

* 会話履歴メモリ統合
* スロット長期保持
* マルチターンPlan更新
* エピソード圧縮
* ユーザー適応型Surface制御
* 並列候補探索
* オンライン辞書拡張
* 軽量ローカル評価器統合

---

## 14. まとめ

LSLM v4 の本質は、
**発散・収束という独自思想を、責務分離された工学構造として成立させること** にある。

v4 で最優先されるべきなのは、応答の見た目の巧さではない。
以下を同時に満たすことである。

* 内部過程が追跡可能
* 各段階が個別に改善可能
* 辞書基盤を中核に据えている
* 外部LLMに依存しすぎない
* 低資源環境でも成立する

この方針を維持する限り、LSLM v4 は
単なる小型チャットボットではなく、
**説明可能で制御可能な小型言語処理アーキテクチャ**として進化できる。
