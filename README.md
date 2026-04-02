# LSLM (Lightweight Small Language Model)

CPU環境でも実用的に動作する小型言語モデル（SLM）を育成するプロジェクト。

大規模言語モデル（LLM）を教師として活用し、  
「生成 → 評価 → 学習」のループによって軽量かつ高性能なモデルを構築する。

---

## 🎯 目的

- CPUで動作可能な実用レベルのSLMを構築
- 特定用途に最適化された軽量モデルの育成
- 継続的に改善可能な学習パイプラインの構築

---

## 🧠 コンセプト

- LLMを教師として使用
- 小型モデルを生徒として育成
- 複数教師による多様な応答生成
- 外部評価モデルによる品質スコアリング
- 蒸留 + 選好学習 + 報酬最適化の段階的学習
- 相対評価（ランキング）を重視
- 失敗データも学習に活用

---

## 🏗 アーキテクチャ

LSLMは以下の5層構造：

1. 入力データ生成層
2. 教師応答生成層
3. 評価・選別層
4. 生徒学習層
5. 検証・運用層

詳細は `architecture.md` を参照

---

## 🤖 使用モデル

### 教師モデル

- `elyza/Llama-3-ELYZA-JP-8B`  
  → 日本語性能・自然さ

- `OpenPipe/Qwen3-14B-Instruct`  
  → 安定性・指示追従

- `LiquidAI/LFM2.5-1.2B-Instruct`  
  → 多様性・軽量生成

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