# LSLM v4 スクリプト棚卸し

## 1. この文書の役割

本書は、リポジトリ内の Python スクリプトを **入口・責務・保守方針** で整理するための文書です。

目的は 3 つです。

1. どのスクリプトを日常的な入口として使うべきかを固定する
2. まとめるべきスクリプトと、分けたままにすべきスクリプトを明文化する
3. 生成物や実験ログと、保守対象コードを混同しないようにする

---

## 2. 先に結論

現時点の v4 は、入口を次の **3 本** に寄せるのが最適です。

1. `main.py`
   - 会話実行と loop-learning の共通入口
2. `tools/lexicon_cli.py`
   - 辞書メンテナンス系の共通入口
3. 個別スクリプト
   - `tools/bootstrap_japanese_lexicon.py`
   - `tools/augment_conversation_lexicon.py`
   - 既存 wrapper 群

つまり、

- **アプリ実行系** は `main.py` に集約する
- **辞書メンテ系** は `tools/lexicon_cli.py` に集約する
- **重いビルド処理** と **狭い専用処理** は個別ファイルを維持する

という形です。

---

## 3. 走査結果の整理

## 3.1 日常的に使うべき入口

### `main.py`

役割:

- `chat`
- `loop-learning`

を切り替えるトップレベル入口。

方針:

- ここにモード切り替えだけを置く
- モード固有ロジックは `src/apps/*` へ逃がす
- `--help` はグローバル案内だけを担当する

### `tools/lexicon_cli.py`

役割:

- 辞書変換
- 辞書逆変換
- 辞書ロードプロファイル
- 会話 seed マージ
- 日本語辞書ブートストラップ

を 1 つの入口へ寄せる。

方針:

- 日常的な辞書メンテは原則ここから呼ぶ
- 既存の個別スクリプトは後方互換 wrapper として残す

---

## 3.2 分けたままでよいもの

### `src/apps/chat_v1.py`

分ける理由:

- 会話実行と対話モードの責務が明確
- `loop_learning_v1.py` と入力源・出力物・実行ライフサイクルが違う

### `src/apps/loop_learning_v1.py`

分ける理由:

- episode 管理
- auto-input
- unknown word overlay
- summary 出力

があり、chat より運用責務が重い

### `tools/bootstrap_japanese_lexicon.py`

分ける理由:

- WordNet / Sudachi / UniDic を使う重いビルド工程
- 日常実行より生成工程のスクリプト
- 引数数も多く、独立していた方が追いやすい

### `tools/augment_conversation_lexicon.py`

分ける理由:

- 会話 seed を既存辞書へ載せる専用処理
- 辞書変換とは責務が違う

---

## 3.3 まとめたほうがよいもの

### 変換・逆変換・プロファイル系

対象:

- `tools/convert_dict_to_binary.py`
- `tools/convert_binary_to_dict.py`
- `tools/profile_lexicon_load.py`

判断:

- いずれも「辞書メンテナンス」という同じ利用文脈で使われる
- 個別に残すと入口が増えすぎる
- ただし内部実装までは無理に 1 ファイルへ統合しない

結論:

- **入口は `tools/lexicon_cli.py` へまとめる**
- 中身は個別ファイルを維持する

これは「実装を無理に合体させないが、使い方はまとめる」という整理です。

---

## 3.4 さらに分けたほうがよいもの

### `src/apps/loop_learning_v1.py`

現状ではやや責務が厚いです。

今後分割候補:

- auto input 生成
- dataset 読込
- unknown word enrichment
- run summary 書き込み

ただし今回は、まず

- 共通 CLI 引数の切り出し
- ロギングと runtime の整理
- 入口の統一

を優先し、内部の大分割は次段階とするのが妥当です。

---

## 3.5 settings に置くべきもの

### `settings/teacher_profiles.yaml`

役割:

- external evaluator / teacher の prompt 管理
- auto-input generator の prompt と payload defaults 管理
- unknown word lexicon enricher の prompt と payload defaults 管理

方針:

- LLM に渡す自然言語本文はここへ集約する
- コード側に英語プロンプト本文を散在させない
- prompt_version はここで上げる

## 4. 不要寄りのもの

### 実行生成物

不要というより **リポジトリ本体へ含め続けるべきではないもの** です。

例:

- `runtime/logs/*`
- `runtime/traces/*`
- `runtime/episodes/*`
- `runtime/learning_runs/*`
- `runtime/cache/*`

これらはソースではなく、実行生成物です。

方針:

- リポジトリ管理対象から外す
- `runtime/README.md` と空ディレクトリだけ残す

### `2024.1python`

用途が実質見えない空ファイルであり、現時点では保守対象として不要です。

---

## 5. 判断を支えた観点

今回の整理は、少なくとも次の観点で整合性を確認しています。

1. CLI の一貫性
2. 初見での発見しやすさ
3. 役割の重複の少なさ
4. 後方互換性
5. 実行系とビルド系の分離
6. runtime 生成物の純度管理
7. ログ・辞書・設定の境界整合
8. 今後の分割余地
9. 低資源運用での扱いやすさ
10. README / docs との一致
11. バグ修正の波及範囲
12. zip 配布時のノイズ削減

---

## 6. 今回の整理結果

### 追加

- `src/apps/cli_common.py`
- `tools/lexicon_cli.py`
- `docs/script_inventory.md`
- `runtime/README.md`

### 更新

- `main.py`
- `src/apps/chat_v1.py`
- `src/apps/loop_learning_v1.py`
- `tools/profile_lexicon_load.py`
- `README.md`
- `architecture.md`
- `docs/docs_guide.md`
- `.gitignore`

### 削除 / 除外

- `2024.1python`
- runtime 配下の過去ログ・trace・episode・cache・summary などの生成物

---

## 7. 今後の優先度

次に手を入れる順番は次が自然です。

1. `loop_learning_v1.py` の内部責務分割
2. relation validation warning の監査出力先整理
3. `tools/` の wrapper に deprecation 案内追加
4. 簡単な smoke test 追加

---

## 8. 結論

現時点の v4 は、

- **アプリ入口は `main.py`**
- **辞書入口は `tools/lexicon_cli.py`**
- **重い生成処理は個別維持**
- **runtime 生成物はリポジトリ本体から分離**

という整理が最も壊れにくく、追いやすい形です。
