# LSLM v4 知識境界設計

## 0. 本書の役割

本書は、LSLM v4 における情報の保存先を定義する。  
目的は、辞書に何でも詰め込んで設計が崩壊することを防ぎ、各情報の責務を明確にすることである。

本書が答える問いは 1 つだけである。

> **この情報は、辞書・実行時・保存・設定のどこに置くべきか。**

思想そのものは `philosophy.md` に、実装順序は `implementation_plan.md` に委譲する。

---

## 1. 前提原則

LSLM v4 では、辞書を次のように定義する。

> **辞書は、ただの知識ではなく、知識のネットワークそのものである。**

したがって、辞書に置くべきなのは、

- ノードとして安定して存在する知識
- エッジとして安定して存在する関係
- 制約として安定して参照される規則

である。  
一方、現在ターンだけに意味がある値、実行結果の記録、実験条件は辞書に置かない。

---

## 2. 4 つの保存先

### 2.1 辞書
比較的安定した知識ネットワーク本体。

特徴:

- 実行のたびに大きく変わらない
- 発散・収束・構造化・表層化の基盤になる
- 人手編集やビルド対象になる
- ノード・エッジ・制約として参照される

### 2.2 実行時
会話や 1 ターンの中だけで意味を持つ一時状態。

特徴:

- ターンごとに変わる
- ワーキングメモリに相当する
- 実行が終われば破棄してよい
- 再利用する場合も、そのまま辞書へ昇格させない

### 2.3 保存
分析・再現・比較・学習のために残す記録。

特徴:

- 実行結果である
- 後で読むために整形される
- 辞書本体ではなく挙動の履歴である
- 再学習や評価の素材になる

### 2.4 設定
実験条件、閾値、モード、外部依存順序などを管理する領域。

特徴:

- コードに埋め込まず差し替えたい
- 環境ごとに変えたい
- 知識そのものではない
- 再現実験に必要である

---

## 3. 判断ルール

新しい情報の置き場に迷ったら、次の順で判定する。

### 3.1 ルール A
その情報は、知識ネットワークのノード・エッジ・制約として**安定的に存在すべきか**。

- Yes → 辞書候補
- No → ルール B へ

### 3.2 ルール B
その情報は、現在のターンや現在の探索だけで意味を持つ**一時状態か**。

- Yes → 実行時
- No → ルール C へ

### 3.3 ルール C
その情報は、後から分析・比較・学習するために**記録として残すべき結果か**。

- Yes → 保存
- No → ルール D へ

### 3.4 ルール D
その情報は、挙動の閾値・実験条件・利用順序のように**差し替え前提の制御値か**。

- Yes → 設定
- No → 設計が曖昧なので再分解する

---

## 4. 辞書に置くもの

### 4.1 Axis
ノードの意味方向を表す連続座標。

例:

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

置き場: **辞書**  
理由: 発散・収束時に繰り返し参照される安定構造だから。

### 4.2 Grammar
局所構文制約、接続条件、文位置制約。

例:

- 品詞
- 文頭可否 / 文末可否
- requires_prev / requires_next
- forbid_prev / forbid_next
- function_word / content_word

置き場: **辞書**  
理由: 表層化や候補フィルタで常に参照する安定制約だから。

### 4.3 Slot の基本フレーム
意味役割や述語骨格の型。

例:

- actor
- target
- recipient
- location
- time
- cause
- state
- predicate_type

置き場: **辞書**  
理由: 役割の型そのものは安定知識であり、現在ターン固有ではないから。

### 4.4 Relation
ノード間の関係ネットワーク。

例:

- synonym
- antonym
- hypernym / hyponym
- associated
- cause_of / caused_by
- part_of / has_part
- often_with
- style_variant

置き場: **辞書**  
理由: 辞書を知識ネットワークとして成立させる中核だから。

### 4.5 Meta / Index
辞書全体のメタ情報と参照用索引。

例:

- schema_version
- build_version
- created_at / updated_at
- alias map
- category map
- search index

置き場: **辞書**  
理由: 整合性維持と高速参照に必要な静的構造だから。

---

## 5. 実行時に置くもの

### 5.1 入力由来の現在値

- raw input
- tokenized input
- input features
- conversation turn context
- 現在時刻から導出した一時値

置き場: **実行時**  
理由: そのターンの処理にのみ必要だから。

### 5.2 Plan の結果

- intent
- response mode
- required_slots
- priority constraints
- fallback policy

置き場: **実行時**  
理由: Plan は現在入力への解釈結果であり、辞書知識ではないから。

### 5.3 探索途中の候補群

- divergence candidates
- candidate scores
- path traces
- convergence survivors
- rejection reasons

置き場: **実行時**  
理由: 現在の探索でのみ意味を持つ中間状態だから。

### 5.4 Slot の充填結果

- filled slots
- unresolved slots
- slot confidence

置き場: **実行時**  
理由: スロットの型は辞書だが、埋まった値は現ターン依存だから。

### 5.5 Surface 生成用の一時構造

- phrasing candidates
- grammar checks in progress
- surface plan
- final response draft

置き場: **実行時**  
理由: 表層化中の作業領域だから。

---

## 6. 保存に置くもの

### 6.1 人間可読ログ

- latest.log
- debug.log
- warning / exception 記録

置き場: **保存**  
理由: 挙動の可観測性を担保する記録だから。

### 6.2 構造化トレース

- 1 ターン 1 JSONL レコード
- 各段階の入力と出力
- timing
- stage scores
- reject reasons

置き場: **保存**  
理由: 再現・比較・学習のための履歴だから。

### 6.3 学習・評価データ

- reward record
- evaluator feedback
- experiment summary
- dataset export
- ablation result

置き場: **保存**  
理由: 実験結果であって辞書知識そのものではないから。

### 6.4 辞書編集の履歴

- build logs
- validation errors
- conversion reports

置き場: **保存**  
理由: 辞書本体ではなく、辞書を作った過程の記録だから。

---

## 7. 設定に置くもの

### 7.1 探索・選別の制御値

- divergence depth
- top_k / beam width
- convergence thresholds
- pruning rules
- retry limits

置き場: **設定**  
理由: 知識ではなく実験条件だから。

### 7.2 スコア・報酬の重み

- stage score weights
- alpha / beta
- reward clipping rules
- evaluator enable flags

置き場: **設定**  
理由: 比較実験で差し替える前提の制御値だから。

### 7.3 ログ関連設定

- console log level
- trace output on/off
- rotation policy
- debug verbosity

置き場: **設定**  
理由: 運用方針であり知識ではないから。

### 7.4 外部依存順序

- `settings/LLM_order.yaml`
- evaluator fallback order
- optional tool usage flags

置き場: **設定**  
理由: 実行環境や運用条件で変わるから。

---

## 8. よくある誤配置

### 8.1 現在ターンの候補を辞書に保存する
誤り。候補群は探索途中の一時状態であり、辞書知識ではない。

### 8.2 閾値を辞書へ入れる
誤り。閾値は知識ではなく実験条件なので設定に置く。

### 8.3 評価結果を辞書へ還元する
そのまま入れるのは誤り。  
まず保存へ記録し、必要なら別プロセスで知識抽出や辞書更新を行う。

### 8.4 実行時状態を丸ごと永続化して辞書扱いする
誤り。永続化しても、それは保存記録であって辞書そのものではない。

---

## 9. 実務上のルール

### 9.1 昇格は段階を踏む
実行時 → 保存 → 検証 → 辞書、の順で扱う。  
一時値を直接辞書へ入れない。

### 9.2 辞書は最小安定単位で保つ
辞書には「安定して何度も参照される知識」だけを残す。

### 9.3 保存は豊かに、辞書は厳しく
迷ったら、まず保存へ残す。  
辞書へ入れるのは十分に安定性と再利用性が確認された後にする。

---

## 10. 結論

LSLM v4 の破綻は、多くの場合「情報の責務が混ざること」から始まる。  
本書の目的は、辞書を肥大化させないことではなく、**辞書を知識ネットワークとして純度高く保つこと** にある。

- 辞書には安定知識を置く
- 実行時には一時状態を置く
- 保存には結果の履歴を置く
- 設定には制御値を置く

この 4 分離を守る限り、v4 は拡張しても崩れにくい。
