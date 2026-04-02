# LSLM (Local Small Language Model)

CPU環境でも実用的に動作する小型言語モデル（SLM）を育成するプロジェクト。

大規模言語モデル（LLM）を教師として活用し、
「生成 → 評価 → 学習」のループによって軽量かつ高性能なモデルを構築する。

---

## 🎯 目的

* CPUで動作可能な実用レベルのSLMを構築
* 特定用途に最適化された軽量モデルの育成
* 継続的に改善可能な学習パイプラインの構築

---

## 🧠 コンセプト

* LLMを教師として使用
* 小型モデルを生徒として育成
* 複数教師による多様な応答生成
* 外部評価モデルによる品質スコアリング
* 蒸留 + 選好学習 + 報酬最適化の段階的学習
* 相対評価（ランキング）を重視
* 失敗データも学習に活用

---

## 🏗 アーキテクチャ

LSLMは以下の5層構造で構成される。

1. 入力データ生成層
2. 教師応答生成層
3. 評価・選別層
4. 生徒学習層
5. 検証・運用層

詳細は以下を参照：

* architecture.md

---

## 🤖 使用モデル

### 教師モデル

* elyza/Llama-3-ELYZA-JP-8B
  日本語性能・自然さ

* OpenPipe/Qwen3-14B-Instruct
  安定性・指示追従

* LiquidAI/LFM2.5-1.2B-Instruct
  多様性・軽量生成

---

### 評価モデル

#### Gemini（Google）

```python
from google import genai

client = genai.Client(http_options={'api_version': 'v1'})

response = client.models.generate_content(
    model='gemini-3-flash-preview',
    contents="Explain how AI works",
)

print(response.text)
```

---

#### OpenAI

```python
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-5.4",
    input="Write a short bedtime story about a unicorn."
)

print(response.output_text)
```

---

### 生徒モデル

* Gemma 3 270M

CPU環境での実用運用を前提とする

---

## 🔁 学習フロー

```
入力生成
   ↓
教師モデルで複数応答生成
   ↓
評価モデルでスコアリング + ランキング
   ↓
データ構築
   ├─ SFTデータ（高品質）
   ├─ 選好データ（ランキング）
   └─ 失敗データ（低品質）
   ↓
学習
   ├─ Phase1: SFT（蒸留）
   ├─ Phase2: 選好学習（DPO / RRHF）
   └─ Phase3: 報酬最適化
   ↓
評価・再ループ
```

---

## 📊 データ設計

保存データ：

* 入力テキスト
* 入力属性（ドメイン / 難易度など）
* 各教師モデルの出力
* 評価スコア（項目別）
* 総合スコア
* ランキング情報
* ペア比較データ
* 採用 / 不採用応答
* メタ情報（生成時間・トークン数など）

---

## 🔥 特徴

### 複数教師蒸留

単一モデルに依存せず、多様な表現を獲得

### 相対評価学習

スコアのブレを回避し、安定した学習を実現

### 失敗データ活用

安全性・一貫性・頑健性を向上

### パイプライン設計

モデル単体ではなくシステム全体を最適化

---

## 🎯 用途特化戦略

SLMは用途特化することで性能が大幅に向上する。

想定用途：

* 日常会話
* 作業支援
* VRChatエージェント
* コマンド生成
* 状況説明・要約

---

## 🚀 今後の展開

* データ生成の自動化
* 評価精度の改善
* DPO / RRHF最適化
* LoRAによる用途別分岐
* CPU推論最適化

---

## 📌 本質

本プロジェクトは単なるモデル開発ではなく、

> 自己改善可能なSLM育成システム

の構築を目的とする。

---

## 🔗 Repository

GitHub:

[https://github.com/nanohana-team/Local-Small-Language-Model](https://github.com/nanohana-team/Local-Small-Language-Model)

---

## 📄 License

TBD

---

## 🤝 Contributing

TBD


©なのは