# Local Small Language Model (LSLM)

軽量モデルをベースに、**複数モデルの会話 + 外部評価 + 反復学習**で性能を引き上げるローカルLLMフレームワーク。

---

## 🚀 概要

本プロジェクトは以下のループで動作します：

1. 複数モデル（teacher + learning）が会話
2. 会話ログを保存
3. Gemini APIで会話を評価
4. learningモデルの発話をデータ化
5. SFT / DPOで再学習
6. 次イテレーションへ

---

## 🧠 コンセプト

* 小型モデルを強くする
* 単発応答ではなく「会話」で鍛える
* 外部モデル（Gemini）を審判として使う
* 蒸留 + 選好学習のハイブリッド

---

## 🗂 ディレクトリ構成

```
project/
├─ main.py
├─ run.bat
├─ README.md
├─ docs/
│  └─ architecture.md
│
├─ src/
│  ├─ chat/        # LLMノードサーバー
│  ├─ loop/        # 会話ループ
│  ├─ inference/   # teacher / learning クライアント
│  ├─ eval/        # Gemini評価
│  ├─ data/        # データ生成・整形
│  └─ train/       # 学習処理
│
├─ config/
│  ├─ teachers.json
│  ├─ learning.json
│  └─ conversation_loop.json
│
├─ data/
│  ├─ prompts/
│  └─ conversations/
│
└─ logs/
```

---

## ▶️ 起動方法

### ワンコマンド起動

```
run.bat
```

または

```
python main.py
```

---

## 🔁 実行フロー

```
[起動]
   ↓
LLMノード起動（別ウィンドウ）
   ↓
会話ループ開始
   ↓
ログ保存
   ↓
Gemini評価
   ↓
SFT/DPOデータ生成
   ↓
learningモデル再学習
   ↓
次ループ
```

---

## 🤖 モデル構成

* Teacher:

  * Qwen系
  * Llama系
  * 軽量モデル（LFMなど）

* Learning:

  * Gemma3 270M（LoRA）

---

## 📊 評価指標（Gemini）

* intent_fit
* naturalness
* consistency
* informativeness
* conciseness
* safety
* completeness
* language_purity

---

## 🔧 今後の拡張

* DPO / RLHF統合
* Vision入力（YOLO連携）
* 音声対話（STT/TTS）
* マルチエージェント強化

---

## ⚠️ 注意

* Gemini APIキーが必要
* ローカルGPU推奨（CPUでも可）
* 初回はモデルDLで時間かかる

---

## ✨ ライセンス

MIT（予定）
