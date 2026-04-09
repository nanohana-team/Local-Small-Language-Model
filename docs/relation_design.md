# LSLM v4 relation 設計

## 1. この文書の役割

本書は、LSLM v4 における **relation の正式な位置づけ** を定義します。  
relation を「辞書の補助情報」と見るか、「辞書の中核」と見るかで、v4 の全設計は変わります。  
本書では、relation を中核と見なした場合の定義・整合性・改善点を固定します。

---

## 2. 先に結論

LSLM v4 では relation を次のように定義します。

> **relation は、語句同士・概念同士の接続可能性と接続意味を定義する場である。**

さらに、v4 の思考過程を次のように定義します。

> **発散とは relation を一定の手順に従って順方向へ接続していく過程である。**  
> **収束とは relation を逆方向または制約方向へたどって接続を絞り込む過程である。**

この定義により、v4 の思考は「曖昧な候補生成」ではなく、**relation を持つグラフ操作** として記述できます。

---

## 3. なぜ relation が重大なのか

## 3.1 語彙だけでは辞書は倉庫に留まる

語が多いだけでは、辞書は単なる保管庫です。  
どの語がどの語へ進めるのかが定義されていなければ、探索も推論も生成も安定しません。

## 3.2 relation があると辞書が思考空間になる

relation があることで初めて、

- どの候補へ広がれるか
- どの候補が近いか
- どの候補が対立するか
- どの述語がどの slot を要求するか
- どの表現へ言い換えられるか

を機械的に扱えます。

## 3.3 relation だけで文章骨格の多くが決まる

適切な relation があれば、

- 主題と述語
- 述語と slot
- 修飾と被修飾
- 丁寧体と常体
- 言い換えと共起

をつなげるだけで、かなりの文章骨格を構成できます。  
そのため relation は、v4 において **語彙以上に重要な構造資産** です。

---

## 4. 10以上の観点から見た整合性

relation 中心思想は、少なくとも次の観点で整合しています。

1. **思想整合**  
   辞書を知識ネットワークと定義するなら、relation を中核に置くのは自然です。
2. **アルゴリズム整合**  
   発散を順方向探索、収束を逆方向・制約探索と定義すると処理意味が明確になります。
3. **データモデル整合**  
   ノードとエッジの分離は concept / relation / slot の責務分離に向いています。
4. **性能整合**  
   relation type ごとの索引を持てば、闇雲な全探索を避けられます。
5. **観測可能性整合**  
   relation path を trace に残せば、ブラックボックス化を防げます。
6. **学習整合**  
   改善対象を relation 追加・削除・重み調整として局所化できます。
7. **安全性整合**  
   source / confidence / review を relation に紐付ければ汚染制御が可能です。
8. **日本語生成整合**  
   意味 relation だけでなく構文 relation と表現 relation を持てば文章生成と噛み合います。
9. **辞書純度整合**  
   恒久 relation と探索履歴を分離すれば辞書の純度を保てます。
10. **実装互換整合**  
    既存の lexical entry 中心 I/O の上に concept-relation 層を追加できます。
11. **品質管理整合**  
    dangling target 禁止や inverse 規約を導入しやすいです。
12. **将来拡張整合**  
    YUNA 系のイベント・状態・意図 relation へも拡張可能です。

結論として、relation 中心思想は **重大であるだけでなく、v4 の他要素と高い整合性を持ちます。**

---

## 5. relation の役割

relation の役割は少なくとも 4 つあります。

### 5.1 探索エッジ

どのノードからどのノードへ進めるかを決めます。

### 5.2 制約エッジ

どの接続が許され、どの接続が弱いかを決めます。

### 5.3 構造エッジ

slot や修飾関係など、文章骨格に必要な接続を保持します。

### 5.4 表現エッジ

言い換え、文体差、共起など、表層化で使う接続を保持します。

---

## 6. relation の最小正式契約

relation には最低限次を持たせます。

```json
{
  "type": "hypernym",
  "target": "concept:consume",
  "weight": 0.92,
  "direction": "outbound",
  "confidence": 0.88,
  "usage_stage": ["divergence", "convergence"],
  "meta": {
    "source": "wordnet",
    "generated": false
  }
}
```

### 必須

- `type`
- `target`

### 強く推奨

- `weight`
- `direction`
- `confidence`
- `usage_stage`
- `meta.source`

### 任意

- `axes`
- `constraints`
- `inverse_type`
- `evidence`
- `meta`

---

## 7. relation type の三層

## 7.1 意味 relation

意味空間の近さや上下関係を表します。

例:

- `synonym`
- `antonym`
- `hypernym`
- `hyponym`
- `cause_of`
- `caused_by`
- `part_of`
- `has_part`

## 7.2 構文 relation

文の骨格を作る接続を表します。

例:

- `predicate_slot`
- `modifier_head`
- `connective_sequence`
- `subject_predicate`
- `argument_role`

## 7.3 表現 relation

表層化やスタイル調整に使う接続を表します。

例:

- `style_variant`
- `politeness_variant`
- `paraphrase`
- `collocation`
- `register_variant`

---

## 8. 発散と収束の relation 的定義

## 8.1 発散

発散は relation を **順方向にたどる探索** です。  
ただし無制限探索ではなく、少なくとも次を持ちます。

- relation type priority
- depth budget
- branching budget
- revisit penalty
- slot / plan による事前制約

## 8.2 収束

収束は relation を **逆方向または制約方向にたどる探索** です。  
少なくとも次の条件を用います。

- plan 適合
- slot 充足
- contradiction 除去
- redundancy 除去
- discourse goal 適合
- relation path の説明可能性

## 8.3 重要な注意

収束は「単純に逆辺をたどるだけ」ではありません。  
slot 制約や plan 制約によって、必要な接続だけを残す過程も収束に含みます。

---

## 9. 品質管理規約

relation を辞書中核に置くなら、次の規約が必須です。

### 9.1 閉じたグラフを基本とする

辞書本体では、`target` が存在しない relation を原則禁止します。  
dangling relation は import review 段階でのみ許容し、本採用前に除去または補完します。

### 9.2 type ごとの方向規約を持つ

例:

- `synonym` → 双方向
- `antonym` → 双方向
- `hypernym` → 片方向
- `hyponym` → 片方向
- `style_variant` → 基本双方向
- `politeness_variant` → 片方向または双方向重み差あり

### 9.3 source と confidence を残す

自動生成 relation と seed relation を区別できなければ品質管理できません。

### 9.4 探索履歴と恒久 relation を混ぜない

今回たどった relation path は trace に残し、辞書本体の relation と混ぜません。

---

## 10. 改善点

relation 中心思想は強いですが、運用を安定させるには次の改善が必要です。

1. **relation schema の正式化**  
   必須項目・方向規約・inverse 規約を固定する。
2. **closed graph validation**  
   dangling target を検出・禁止する strict mode を常設する。
3. **relation taxonomy の固定**  
   意味・構文・表現の三層を正式採用する。
4. **relation index の拡張**  
   type 別、target 別、inverse 別の索引を持つ。
5. **trace の強化**  
   explored path と accepted path を分けて残す。
6. **score の強化**  
   relation coverage / precision / dangling rate を評価軸へ追加する。
7. **import review の導入**  
   外部資源由来 relation を staged review で昇格させる。
8. **slot との接続強化**  
   predicate-slot 系 relation を slot_frame と整合させる。

---

## 11. 実装判断の簡易ルール

relation を追加したくなったら、次の順で問います。

1. その接続は安定知識か
2. 接続の向きは明確か
3. target は辞書内に存在するか
4. type は三層 taxonomy のどれに属するか
5. source と confidence を残せるか
6. 発散・収束のどこで使うか明確か

1 つでも答えられない場合は、review へ送るのが安全です。

---

## 12. 結論

LSLM v4 において relation は、
**辞書の補助情報ではなく、思考と生成を成立させる接続規則の中心** です。

この定義により、

- 発散は relation の順方向探索
- 収束は relation の逆方向または制約探索
- 文章骨格は relation の組み合わせ
- 改善対象は relation の品質

として扱えるようになります。

つまり v4 は、語彙を積み上げるだけの系ではなく、  
**relation を鍛えることで強くなる系** です。
