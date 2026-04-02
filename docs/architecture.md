# LSLM Architecture

---

## 🧠 全体構造

```
         ┌────────────┐
         │  Prompt    │
         └────┬───────┘
              ↓
     ┌──────────────────┐
     │ Conversation Loop│
     └────┬─────────────┘
          ↓
 ┌───────────────────────┐
 │ Multi-Model Dialogue  │
 │                       │
 │ Teacher A             │
 │ Teacher B             │
 │ Teacher C             │
 │ Learning Model        │
 └─────────┬─────────────┘
           ↓
    ┌──────────────┐
    │ Session Log  │
    └────┬─────────┘
         ↓
 ┌──────────────────────┐
 │ Gemini Evaluation    │
 └────┬─────────────────┘
      ↓
 ┌──────────────────────┐
 │ Dataset Builder      │
 │  - SFT               │
 │  - DPO               │
 └────┬─────────────────┘
      ↓
 ┌──────────────────────┐
 │ Training Pipeline    │
 │  - LoRA              │
 └────┬─────────────────┘
      ↓
 ┌──────────────────────┐
 │ Updated Model        │
 └─────────┬────────────┘
           ↓
      Next Iteration
```

---

## 🔁 Conversation Loop

各セッションは以下の順で進行：

```
Teacher A
→ Teacher B
→ Teacher C
→ Learning
→ (repeat)
```

特徴：

* 同一セッションID
* コンテキスト共有
* 最大ターン数制御

---

## 🧾 Sessionデータ構造

```
{
  session_id,
  prompt_id,
  input,
  turns: [
    {
      turn_index,
      speaker,
      model,
      text,
      latency_sec
    }
  ]
}
```

---

## 📊 Evaluation

Geminiが行う処理：

* モデルランキング
* learning発話のスコアリング
* preferenceペア生成

---

## 🧪 Dataset生成

### SFT

* 高スコアlearning発話のみ採用

### DPO

* teacher vs learning の比較ペア生成

---

## 🧠 Training

* Base: Gemma3 270M
* 方法: LoRA
* 継続学習対応

---

## ⚙️ 設計思想

### 1. 小型モデル強化

巨大モデルに頼らず性能を引き上げる

### 2. 会話ベース学習

単発ではなく文脈込み

### 3. 外部評価

自己評価を排除

### 4. 反復改善

ループで性能向上

---

## 🔥 ボトルネック

* Gemini API速度
* 学習時間
* データ品質

---

## 🚀 将来拡張

* RLHF統合
* マルチモーダル
* 分散学習
* 自己進化ループ

---

## 💡 まとめ

LSLMは：

「軽量モデル × 会話 × 外部評価 × 反復学習」

で進化するシステム
