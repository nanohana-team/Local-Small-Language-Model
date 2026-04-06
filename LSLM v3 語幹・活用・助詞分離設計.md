# 語幹・活用・助詞分離設計まとめ（LSLM v3）

## 1. 設計の核心

本設計では、日本語処理を以下の4要素に完全分離する。

* 語幹（Semantic Core）
* 活用（Inflection / State Transformer）
* 助詞（Relation Operator）
* 表層変換（Surface Realization）

---

## 2. 基本思想

### 2.1 語幹

* 意味の核
* Axis / Slot / Relation を保持
* 発散・収束の対象

👉 語幹 = 思考ノード

---

### 2.2 活用

* 意味は持たない
* 文の状態を変更する

例：

* 時制（過去）
* 相（進行）
* モダリティ

👉 活用 = 状態遷移オペレータ

---

### 2.3 助詞

* 語と語の関係を定義

例：

* が → 主語
* を → 目的語

👉 助詞 = 関係オペレータ

---

### 2.4 表層変換

* 内部構造から自然文を生成
* 活用による語形変化を適用

👉 表層 = 最終レンダリング

---

## 3. データ構造

### 3.1 語幹

```json
{
  "lemma": "書く",
  "stem_id": "kak",
  "type": "stem",
  "grammar": {
    "pos": "verb",
    "conj_class": "godan_ku"
  },
  "vector": {},
  "slots": [],
  "relations": []
}
```

---

### 3.2 活用

```json
{
  "word": "た",
  "type": "inflection",
  "grammar": {
    "pos": "auxiliary",
    "requires_prev": ["verb_stem"],
    "can_end": true
  },
  "effect": {
    "tense": "past"
  }
}
```

---

### 3.3 助詞

```json
{
  "word": "が",
  "type": "particle",
  "effect": {
    "mark_role": "actor"
  }
}
```

---

## 4. 活用変形の扱い

### 問題

語幹の一部が変化する（例：書く→書き）

---

### 解決

語幹は変化しないものとして扱う

👉 表層変換で変形を行う

---

### 内部表現

```json
{
  "stem_id": "kak",
  "conj_class": "godan_ku"
}
```

---

### 活用ルール

```json
{
  "godan_ku": {
    "mizen": "か",
    "renyo": "き",
    "shushi": "く",
    "katei": "け"
  }
}
```

---

## 5. 処理フロー

1. 入力をトークン化
2. 表層 → lemma / stem_id に正規化
3. Semantic Recall（語幹のみ）
4. Slot Filling（語幹のみ）
5. 状態適用（活用 effect）
6. 関係付与（助詞）
7. 表層生成（活用 + 接続）

---

## 6. 状態モデル

```python
state = {
  "tense": "past",
  "aspect": "progressive"
}
```

---

## 7. 設計ルール

### やってはいけない

* 活用に意味ベクトルを持たせる
* 語幹に時制を持たせる
* 表層形を辞書の主キーにする

---

### 必須ルール

* stem_id で意味の同一性を管理
* 活用は effect として定義
* 表層は最後に生成

---

## 8. 結論

* 語幹は変わらない
* 活用が表層を変える
* 助詞が関係を作る

👉 日本語は「意味 + 状態遷移 + 関係 + 表層変換」で構成される
