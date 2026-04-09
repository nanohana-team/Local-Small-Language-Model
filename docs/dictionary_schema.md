# LSLM v4 辞書スキーマ設計

## 1. この文書の役割

本書は、LSLM v4 における辞書の中心仕様を定義します。  
思想上の中心である **concept ベースの知識ネットワーク** と、現在の実装資産である **lexical entry ベースの互換コンテナ** をどう接続するかまで含めて整理します。

relation 自体の思想と規約は `relation_design.md` を主参照とし、本書では relation を **辞書スキーマへどう載せるか** に集中します。

---

## 2. まず結論

v4 の辞書は次の二層で考えます。

### 2.1 意味の中心

- `concept`
- `sense`
- `relation`
- `slot_frame`
- `axis`

### 2.2 保存と実装の中心

- `lexical_entry`
- `surface_forms`
- `indexes`
- `meta`

つまり、**意味の中心は concept + relation**、**保存上の基本単位は lexical entry** です。

---

## 3. なぜ二層に分けるのか

辞書を lexical entry だけで持つと、表記中心の設計になりやすく、意味分岐や概念関係が痩せます。  
逆に concept だけで持つと、活用形や文法制約や表記ゆれの扱いが不便になります。

そのため v4 では、

- **思考の中心** は concept と relation に置く
- **保存と実装互換** は lexical entry を土台にする

という分担を採ります。

---

## 4. 中核要素

## 4.1 concept

意味ネットワークの中心ノードです。  
「何を意味しているか」の本体は concept が持ちます。

例:

- 犬
- 動物
- 食べる
- 喜び
- 移動

concept が持つ代表情報:

- `id`
- `label`
- `category`
- `description`
- `axes`
- `relations`
- `default_slot_frame_id`
- `meta`

---

## 4.2 lexical_entry

辞書ファイル上の基本レコードです。  
語としての見た目・活用・文法制約を持ちます。

例:

- 「犬」
- 「食べる」
- 「走る」

lexical entry が持つ代表情報:

- `id`
- `lemma`
- `reading`
- `surface_forms`
- `grammar`
- `senses`
- `style_tags`
- `frequency`
- `meta`

---

## 4.3 sense

1 つの lexical entry が複数の意味を持つ場合の分岐単位です。  
各 sense は 1 つ以上の concept に接続されます。

例:

- はし → 橋 / 箸
- かける → 掛ける / 駆ける / 電話を掛ける

sense が持つ代表情報:

- `id`
- `gloss`
- `concept_ids`
- `slot_frame_override`
- `usage_notes`
- `priority`

---

## 4.4 surface_form

表面形・活用形・表記ゆれの集合です。  
検索・分割・表層化で使います。

例:

- 食べる
- 食べた
- 食べます
- たべる

---

## 4.5 relation

relation は主に concept 同士を結ぶ **接続規則付きエッジ** です。  
ただし一部は sense や lexical entry を橋渡しする形で補助的に持つこともあります。

### relation の正式な意味

- どのノードからどのノードへ進めるか
- その接続が何を意味するか
- どの段階でその接続を使うか
- どの程度その接続を信頼するか

をまとめて表します。

### relation の最小正式契約

relation は最低限次を持ちます。

- `type`
- `target`

強く推奨する追加情報:

- `weight`
- `direction`
- `confidence`
- `usage_stage`
- `meta.source`

任意補助:

- `axes`
- `constraints`
- `inverse_type`
- `evidence`
- `meta`

### relation の三層 taxonomy

- **意味 relation**  
  `synonym`, `antonym`, `hypernym`, `hyponym`, `cause_of`, `caused_by`, `part_of`, `has_part`
- **構文 relation**  
  `predicate_slot`, `modifier_head`, `connective_sequence`, `subject_predicate`, `argument_role`
- **表現 relation**  
  `style_variant`, `politeness_variant`, `paraphrase`, `collocation`, `register_variant`

### relation の規約

- 辞書本体では `target` が存在しない relation を原則禁止する
- `synonym` / `antonym` / `style_variant` は双方向規約を持てる
- `hypernym` / `hyponym` / `part_of` / `has_part` / `cause_of` / `caused_by` は片方向規約を持つ
- 発散と収束の双方に使う relation だけでなく、片方専用 relation を許す
- relation path と relation 本体を混同しない

---

## 4.6 slot_frame

述語的な concept / sense が要求する意味役割の枠です。

代表例:

- actor
- target
- recipient
- location
- time
- cause
- result
- manner

slot_frame が持つ代表情報:

- `id`
- `slots`
- `required`
- `constraints`
- `examples`

---

## 4.7 grammar

語の接続制約や振る舞いを表す集合です。  
基本は lexical entry に置き、必要なら sense ごとに例外上書きを許します。

代表例:

- `pos`
- `sub_pos`
- `conjugation_type`
- `conjugation_slot`
- `connectability`
- `independent`
- `can_start`
- `can_end`
- `content_word`
- `function_word`
- `requires_prev`
- `requires_next`
- `forbid_prev`
- `forbid_next`

---

## 4.8 axis

意味方向の連続座標です。  
主に concept に属し、relation に補助的に持たせても構いません。

初期候補:

- valence
- arousal
- abstractness
- sociality
- temporality
- agency
- causality
- certainty
- deixis
- discourse_force

---

## 5. 現行実装との互換点

現行 `src/core/io/lsd_lexicon.py` は、主に次の形を正規化対象として扱います。

```json
{
  "meta": {},
  "entries": {
    "食べる": {
      "word": "食べる",
      "category": "verb",
      "hierarchy": ["content_words", "verbs", "stems"],
      "vector": {
        "valence": 0.1,
        "agency": 0.8
      },
      "grammar": {
        "pos": "verb",
        "sub_pos": "independent",
        "conjugation_type": "ichidan",
        "conjugation_slot": "dictionary",
        "connectability": 0.9,
        "independent": true,
        "can_start": false,
        "can_end": true,
        "content_word": true,
        "function_word": false,
        "roles": ["predicate"],
        "requires_prev": [],
        "requires_next": [],
        "forbid_prev": [],
        "forbid_next": []
      },
      "slots": [],
      "relations": [],
      "frequency": 0.42,
      "style_tags": ["neutral"],
      "meta": {}
    }
  },
  "indexes": {}
}
```

この形式は **互換レイヤ** として今後も重要です。  
ただし、これだけでは concept と relation の扱いが弱いため、v4 の本命仕様としては不足があります。

---

## 6. relation 中心設計との橋渡し

現行互換形式から relation 中心設計へ移るときの橋渡しは次の通りです。

1. lexical entry に `senses` と `concept_ids` を追加する
2. concept に relation を集約して保持できるようにする
3. `slot_frame` を独立定義し、構文 relation と接続する
4. index を relation 探索前提へ拡張する

ここで重要なのは、**lexical entry を壊さずに relation 層を厚くする** ことです。

---

## 7. 目標スキーマ

v4 が最終的に目指す辞書コンテナは、次のような概形です。

```json
{
  "meta": {
    "schema_version": "v4",
    "build_version": "0.0.0",
    "semantic_axes": [
      "valence",
      "arousal",
      "abstractness",
      "sociality",
      "temporality",
      "agency",
      "causality",
      "certainty",
      "deixis",
      "discourse_force"
    ]
  },
  "concepts": {
    "concept:eat": {
      "id": "concept:eat",
      "label": "食べる",
      "category": "event",
      "description": "食物を摂取する行為",
      "axes": {
        "agency": 0.8,
        "causality": 0.4,
        "abstractness": -0.3
      },
      "relations": [
        {
          "type": "hypernym",
          "target": "concept:consume",
          "weight": 0.92,
          "direction": "outbound",
          "confidence": 0.95,
          "usage_stage": ["divergence", "convergence"],
          "meta": {
            "source": "seed"
          }
        }
      ],
      "default_slot_frame_id": "slot_frame:eat",
      "meta": {}
    }
  },
  "slot_frames": {
    "slot_frame:eat": {
      "id": "slot_frame:eat",
      "slots": [
        {"name": "actor", "required": true},
        {"name": "target", "required": true},
        {"name": "location", "required": false},
        {"name": "time", "required": false}
      ]
    }
  },
  "lexical_entries": {
    "lex:食べる": {
      "id": "lex:食べる",
      "lemma": "食べる",
      "reading": "たべる",
      "surface_forms": [
        {"text": "食べる", "kind": "lemma"},
        {"text": "食べた", "kind": "past"},
        {"text": "食べます", "kind": "polite"}
      ],
      "grammar": {
        "pos": "verb",
        "sub_pos": "independent",
        "conjugation_type": "ichidan",
        "conjugation_slot": "dictionary",
        "connectability": 0.9,
        "independent": true,
        "can_start": false,
        "can_end": true,
        "content_word": true,
        "function_word": false,
        "requires_prev": [],
        "requires_next": [],
        "forbid_prev": [],
        "forbid_next": []
      },
      "senses": [
        {
          "id": "sense:食べる:1",
          "gloss": "食物を摂取する",
          "concept_ids": ["concept:eat"],
          "priority": 1.0
        }
      ],
      "style_tags": ["neutral"],
      "frequency": 0.42,
      "meta": {}
    }
  },
  "indexes": {
    "concept_to_entries": {
      "concept:eat": ["lex:食べる"]
    },
    "surface_to_entry": {
      "食べる": ["lex:食べる"],
      "食べた": ["lex:食べる"]
    },
    "relation_by_type": {
      "hypernym": ["concept:eat"]
    },
    "relation_target_to_sources": {
      "concept:consume": ["concept:eat"]
    }
  }
}
```

---

## 8. 移行方針

### 8.1 すぐに全部移行しない

現行 I/O が安定資産なので、最初から破壊的変更はしません。

### 8.2 当面の橋渡し方針

現行 `entries[*]` に、必要に応じて以下を追加できるようにします。

- `concept_ids`
- `senses`
- `surface_forms`
- `slot_frame_id`
- `relation_paths`

これにより、互換性を残しつつ relation 中心構造へ近づけます。

### 8.3 段階的移行の順序

1. 現行 entry 形式に `senses` を追加
2. `concept_ids` を sense 側へ追加
3. `slot_frame` を独立定義化
4. `concepts` セクションを追加
5. `indexes` を relation 基準へ拡張
6. 実行時処理の主参照を concept + relation 側へ移行

---

## 9. どこを辞書に入れるべきか

辞書に入れるべきもの:

- 安定した概念構造
- 語彙と表記ゆれ
- 語義分岐
- relation
- 文法制約
- 意味役割の型
- 検索と探索のための索引

辞書に入れるべきでないもの:

- 今回のターンだけで作った候補
- その場のスコア実測値
- ログレコード
- 実験しきい値
- LLM 使用順序

---

## 10. 設計上の注意

### 10.1 concept を省略しない

実装都合だけで lexical entry 中心へ閉じると、意味ネットワークとして痩せます。

### 10.2 sense を後回しにしすぎない

多義語を lexical entry 直結で雑に持つと、後で収束ロジックが破綻しやすくなります。

### 10.3 relation を文字列タグで終わらせない

relation は探索のエッジです。  
種類・向き・重み・信頼度・利用段階を持てる設計にしておくべきです。

### 10.4 dangling relation を放置しない

target が存在しない relation を辞書本体に残すと、探索が空振りだらけになります。

### 10.5 grammar を Surface だけの都合にしない

grammar は表層化だけでなく、分割・候補選別・接続制約にも使います。

### 10.6 indexes は辞書外キャッシュではなく辞書構成要素として扱う

高速性は v4 の重要要件なので、索引は正式な辞書要素に含めます。

---

## 11. 結論

v4 の辞書設計は、
**concept と relation を意味の中心に置きつつ、lexical entry を保存互換レイヤとして使う二層構造** が最適です。

これにより、

- いまある辞書 I/O 資産を活かせる
- 多義語・概念関係・意味役割を正しく伸ばせる
- 発散・収束・スロット充填を relation 中心で組みやすい

という 3 つを両立できます。
