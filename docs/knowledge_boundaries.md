# LSLM v4 知識境界設計

## 1. この文書の役割

本書は、LSLM v4 における情報の置き場を定義します。  
設計で一番壊れやすいのは「便利だから」と情報を 1 か所へ寄せ集めることです。  
この文書の目的は、その混線を防ぐことにあります。

---

## 2. 先に結論

LSLM v4 では、情報の置き場を少なくとも次の 4 種類に分けます。

```text
1. 辞書     = 安定知識
2. 実行時   = 現ターンの状態
3. 保存     = 実行結果の記録
4. 設定     = 実験条件と方針
```

この 4 分離は v4 の中核ルールです。

---

## 3. 大前提

LSLM v4 では辞書を次のように定義します。

> **辞書は、ただの知識ではなく、知識のネットワークそのものである。**

したがって辞書に入るべきなのは、
**安定して再利用される知識構造** です。  
逆に、ターン依存の一時値や、実行後に初めて生まれるログや、運用しきい値は辞書に入れません。

relation 中心設計では特に、**relation 本体** と **relation path** と **relation review 候補** を混ぜないことが重要です。

---

## 4. 各領域の定義

## 4.1 辞書

### 定義

再利用可能な知識ネットワーク。

### 入れるもの

- concept
- lexical entry
- sense
- relation
- slot frame
- grammar
- axis
- alias / surface index
- category / hierarchy index

### 入れないもの

- 今回の入力から得た候補
- 一時スコア
- ログ本文
- 実験しきい値
- 外部 API 順序
- review 未昇格 relation

### 判断基準

「次の実行でも、そのまま知識として使えるか」で判断します。

---

## 4.2 実行時

### 定義

1 回の入力処理中だけ意味を持つ一時状態。

### 代表例

- input_features
- plan
- divergence candidates
- convergence candidates
- explored relation candidates
- accepted relation candidates
- filled slots
- surface plan
- temporary working memory
- current conversation turn state

### 重要な性質

- ターンごとに変わる
- 失っても知識自体は壊れない
- 保存する場合も、そのまま辞書へ戻してはいけない

---

## 4.3 保存

### 定義

実行後に残す記録。  
知識そのものではなく、**挙動の履歴** を扱います。

### 代表例

- runtime logs
- trace jsonl
- reward records
- evaluator feedback
- learning episodes
- benchmark results
- profiling results
- relation import review
- relation audit report

### 重要な性質

- 再現実験や分析に使う
- 必要なら辞書改善の材料になる
- ただし直接辞書へ混入させない

---

## 4.4 設定

### 定義

実行方針・順序・しきい値・外部依存を管理する領域。

### 代表例

- LLM 使用順序
- divergence 深さ
- convergence 採用数
- relation type priority
- trace 出力方針
- reward 重み
- 実験プロファイル

### 重要な性質

- コードに埋め込まない
- 実験条件ごとに差し替え可能にする
- 辞書知識とは切り離す

---

## 5. 迷いやすい要素の置き場

## 5.1 unknown word

- **入力ごとに検出された unknown word** → 実行時
- **検討の結果、辞書へ正式採用した語** → 辞書
- **unknown word 検出ログ** → 保存

## 5.2 reward

- **今回の応答に対する score / reward 実測値** → 保存
- **reward の計算式や重み** → 設定
- **評価に使う stable slot frame や grammar** → 辞書

## 5.3 会話履歴

- **現セッションの作業用履歴** → 実行時
- **分析のために切り出した会話ログ** → 保存
- **永続的な人格設定や方針** → 設定または別メモリ層

## 5.4 relation

- **辞書に元々ある relation** → 辞書
- **今回たどった relation の経路** → 保存
- **今回だけ採用した relation 候補** → 実行時
- **import 時の未昇格 relation 候補** → 保存
- **external teacher が返した teacher_hints / teacher_target** → 保存（review candidate）
- **昇格後の relation** → 辞書

---

## 6. 辞書へ還元してよい条件

保存領域の情報を辞書へ戻してよいのは、少なくとも次を満たす場合です。

1. 一過性ではない
2. 複数回の観測や検証で再利用価値がある
3. 辞書構造として表現できる
4. 由来が追跡できる
5. 追加後の整合性検証が通る

つまり「便利そうだから記録をそのまま辞書へ入れる」は禁止です。

relation で言えば、review に残った接続候補をそのまま本辞書へ入れてはいけません。  
source / confidence / target 存在性を確認したものだけを昇格させます。

---

## 7. アンチパターン

### 7.1 辞書にログを混ぜる

最悪です。  
知識ネットワークと挙動履歴が混ざり、意味構造が壊れます。

### 7.2 設定値を辞書メタへ押し込む

辞書は知識であって実験条件ではありません。

### 7.3 一時候補を辞書へそのまま反映する

発散候補や収束候補は、あくまで runtime state です。

### 7.4 保存データを辞書と同列に扱う

履歴は履歴であり、知識そのものではありません。

### 7.5 review 未昇格 relation を本採用扱いする

後で relation 汚染が起き、発散と収束が腐ります。

---

## 8. 実装判断の簡易ルール

新しい情報を追加したくなったら、次の順に問います。

1. 次回実行でも知識として再利用するか
2. 今回の処理だけに必要か
3. 後で分析するために残すべきか
4. 単なる方針・重み・順序か

対応先はそれぞれ次です。

- 1 → 辞書
- 2 → 実行時
- 3 → 保存
- 4 → 設定

---

## 9. 現在のリポジトリへの対応

現時点で確認できる代表的な対応は次の通りです。

- `src/core/io/lsd_lexicon.py` が扱う辞書コンテナ → 辞書
- `tools/convert_dict_to_binary.py` の変換結果 → 辞書成果物
- `settings/LLM_order.yaml` → 設定
- 今後の JSONL trace / learning episodes → 保存
- 今後の Plan / divergence candidates / slots → 実行時
- 今後の relation review / relation audit → 保存

---

## 10. 結論

LSLM v4 の設計で最も重要なのは、
**辞書を知識ネットワークとして純度高く保ち、relation 本体・relation path・review 候補・設定値を混ぜないこと** です。

この境界が守られている限り、

- 辞書は太っても壊れにくい
- ログは豊富でも汚染しない
- 実験条件を変えても比較できる
- 失敗箇所を切り分けやすい

という v4 の強みが保たれます。
