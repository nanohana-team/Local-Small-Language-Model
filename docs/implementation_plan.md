# LSLM v4 実装計画

## 1. この文書の役割

本書は、LSLM v4 を **壊れにくい順番で実装するための計画書** です。  
「作れそうなものから作る」のではなく、後戻りコストが低く、責務分離を崩しにくい順番を定義します。

---

## 2. 実装方針の結論

v4 は次の順序で進めるのが最適です。

1. 文書とスキーマを固定する
2. 辞書基盤を完成させる
3. relation 基盤を完成させる
4. ログ基盤を常設する
5. Plan を最小実装する
6. Divergence / Convergence を通す
7. Slot / Surface を通す
8. 内部評価を入れる
9. 外部 evaluator を補助的に接続する
10. 学習ループを閉じる

理由は単純で、**辞書・relation・観測・責務分離がない状態で学習へ進むと、何が悪かったのか分からなくなる** からです。

---

## 3. 現在の前提資産

v4 で既に持っている資産:

- `src/core/io/lsd_lexicon.py`
- `tools/convert_dict_to_binary.py`
- `settings/LLM_order.yaml`
- `docs/*.md`

このため、v4 の出発点は「空」ではありません。  
**辞書基盤と設計仕様は先にある** ので、そこを軸に縦へ通すのが正解です。

---

## 4. フェーズ分割

## Phase 0: 仕様固定

### 目的

後で責務が混線しないように、最上位仕様を先に確定する。

### 作業

- README 更新
- architecture 更新
- docs 再編
- relation 定義の明文化
- 辞書スキーマの明文化
- 境界設計の明文化

### 完了条件

- docs の役割分担が明確
- concept / lexical entry / sense / relation の定義が固定
- 辞書・実行時・保存・設定の境界が固定

---

## Phase 1: 辞書基盤の完成

### 目的

辞書ネットワークを、読み書き・正規化・変換の面で安定化する。

### 作業

- 現行 entry 形式のバリデーション強化
- `senses` / `concept_ids` 互換拡張
- `slot_frame` の表現追加
- `indexes` の拡張方針確定
- LSD / LSDX 互換性維持

### 完了条件

- JSON / LSD / LSDX の相互変換が安定
- 互換 entry 形式から target schema への橋渡しが定義済み
- 最低限の concept 参照を辞書上で持てる

---

## Phase 2: relation 基盤の完成

### 目的

relation を辞書の補助情報ではなく、中核接続規則として安定化する。

### 作業

- relation schema の正式化
- relation taxonomy の固定
- relation index の追加
- inverse 規約の固定
- closed graph validation の実装
- import review からの relation 昇格フロー定義

### 完了条件

- relation type / direction / usage_stage が明確
- dangling relation を strict mode で禁止できる
- relation by type / target の索引が安定
- 外部由来 relation を review 経由で昇格できる

---

## Phase 3: ログ基盤の常設

### 目的

後工程の分析と学習のために、観測を先に成立させる。

### 作業

- runtime log 設計
- trace jsonl 設計
- turn_id / session_id 規約
- explored path / accepted path の分離
- 例外・warning の集約
- timing 計測

### 完了条件

- 1 入力 = 1 trace が確実に残る
- 中間段階を欠落なく追跡できる
- ログを見れば失敗箇所の当たりが付く

---

## Phase 4: Plan v1

### 目的

入力から応答方針を決める最小層を作る。

### 作業

- 発話タイプ分類
- intent 決定
- required_slots 決定
- relation type priority の初期決定
- fallback plan 定義

### 完了条件

- どんな入力にも最低限の plan が返る
- 入力に対し plan が空にならない
- Slot 側が plan を参照できる

---

## Phase 5: Divergence / Convergence v1

### 目的

辞書ネットワークを実際に思考へ使い始める。

### 作業

- 起点語抽出
- relation 展開
- relation priority 適用
- depth / branching budget 適用
- axis 近傍探索
- category / hierarchy 制約
- 候補採用 / 棄却の初期 scoring

### 完了条件

- 発散候補が複数安定生成される
- 収束で採用候補が説明可能に選ばれる
- relation path が trace に残る
- 発散と収束が relation 操作として説明できる

---

## Phase 6: Slot / Surface v1

### 目的

意味骨格から最終応答までを最短で通す。

### 作業

- slot frame 決定
- predicate-slot relation の参照
- 必須 slot 充填
- 未充填 slot の可視化
- テンプレートベース表層化
- 文体調整

### 完了条件

- 短文会話で意味破綻の少ない応答が出る
- slot と表層の責務が分離されている
- Surface が新規意味を足していない

---

## Phase 7: 内部評価

### 目的

学習前に、構造の良し悪しを内部から測れるようにする。

### 初期評価軸

- plan_fitness
- relation_coverage
- divergence_relevance
- convergence_fitness
- relation_precision
- slot_fitness
- grammar_fitness
- input_retention
- dangling_rate

### 完了条件

- trace に内部 score が残る
- どの段階が弱いか数値で見える
- relation 品質の悪化を検知できる

---

## Phase 8: 外部 evaluator 接続

### 目的

自然さや有用性の信号を補助的に得る。

### 作業

- evaluator adapter 実装
- `settings/LLM_order.yaml` 参照
- internal / external の分離保存
- feedback_text の保存

### 完了条件

- evaluator 不在でも動く
- evaluator ありなら external score を追加取得できる
- 外部評価が中核ロジックを侵食しない

---

## Phase 9: 学習ループ閉鎖

### 目的

辞書改善・方針更新・比較実験を回せる状態にする。

### 作業

- episode record 保存
- 採用 / 棄却の理由保存
- 改善候補抽出
- unknown word の辞書昇格フロー
- relation 昇格・降格フロー
- benchmark 回帰確認

### 完了条件

- 何を変えたら何が良くなったか追跡できる
- 辞書改善とモデル改善が分離されている

---

## 5. 最初の縦スライス

v4 の最初の完成形は、豪華である必要はありません。  
以下が通れば十分です。

1. 入力を受け取る
2. Plan を作る
3. relation priority を決める
4. 辞書から relation をたどって候補を広げる
5. relation を絞って候補を残す
6. slot を埋める
7. 1〜3 文で返す
8. trace を残す

この最小縦スライスが通る前に、複雑な学習や高度な会話履歴へ進まないことが重要です。

---

## 6. 優先度の判断基準

新規タスクの優先順位は、次の順で判断します。

1. 境界を守るか
2. relation 品質を守るか
3. 観測可能性を上げるか
4. 辞書中心性を強めるか
5. 最小縦スライスを通すか
6. 品質改善に寄与するか
7. コスト増を抑えられるか

この順番を逆にすると、短期的には動いても後で壊れます。

---

## 7. やってはいけない順序

- 先に RL を始める
- 先に長文生成へ広げる
- 先に会話履歴を肥大化させる
- 先に外部 LLM 依存を深くする
- 先に Surface を賢くする
- 先に relation を曖昧なまま増やす

理由は、どれも **内部構造が未固定のまま外側だけ派手になる** からです。

---

## 8. 現時点の次手

現行リポジトリから見た、最も自然な次手は次です。

1. relation schema のコード化
2. relation closed-graph validation
3. relation index の追加
4. trace 設計のコード化
5. Plan v1 の実装
6. Divergence / Convergence の最小接続

この順序なら、既存資産を最大限活かしながら v4 の中核へ入れます。

---

## 9. 結論

v4 の実装計画は、
**辞書 → relation → 観測 → Plan → 発散収束 → Slot / Surface → 評価 → 学習** の順が最適です。

この順番は地味ですが、

- 後戻りが少ない
- 失敗箇所が分かる
- relation 中心思想を守れる
- 低資源設計を維持できる

という意味で、最も壊れにくい進め方です。
