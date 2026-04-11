# LSLM v4 Architecture

## 1. この文書の役割

この文書は、LSLM v4 の **処理責務とリポジトリ配置** を対応づける地図です。

- `docs/philosophy.md` は何を目指すか
- `docs/relation_design.md` は relation をどう定義するか
- `docs/dictionary_schema.md` は辞書をどう定義するか
- 本書はそれを **どのファイルが担うか** に落とします

---

## 2. システム全体像

```text
User Input
   ↓
Input Analysis
   ↓
Plan
   ↓
Divergence
   ↓
Convergence
   ↓
Slot Filling
   ↓
Surface Rendering
   ↓
Response
```

この縦パイプラインを、次の横断層が支えます。

- Dictionary / Relation
- Logging / Trace
- Scoring / Reward
- External evaluator / teacher
- Runtime records
- Settings / Policy

---

## 3. 入口の責務

## 3.1 `main.py`

トップレベル入口です。

責務:

- `chat`
- `loop-learning`

のモード選択だけを行います。

ここに重いロジックは持たせません。

## 3.2 `src/apps/chat_v1.py`

短い対話実行と single-turn 実行を担当します。

責務:

- エンジン起動
- 1 turn 実行
- interactive chat
- trace 記録

## 3.3 `src/apps/loop_learning_v1.py`

学習エピソード収集を担当します。

責務:

- dataset / inline / auto-input の収集
- episode 単位の turn 実行
- unknown word overlay 更新
- episode record 保存
- run summary 保存

## 3.4 `src/apps/cli_common.py`

CLI 共通引数と trace mode 解決の薄い共通層です。

責務:

- chat / loop-learning 間で共通の engine 引数を揃える
- `--debug` と `trace_mode` の整合を保つ

---

## 4. `src/core/` の責務

### `src/core/io/`

辞書 I/O と正規化。

主役:

- `lsd_lexicon.py`

### `src/core/relation/`

relation schema / index / validation。

### `src/core/planning/`

Plan 生成。

### `src/core/divergence/`

入力解析と relation 発散。

### `src/core/convergence/`

候補収束と採用判定。

### `src/core/slotting/`

意味役割充填。

### `src/core/surface/`

表層化。

### `src/core/scoring/`

内部構造評価と reward 計算。

### `src/core/logging/`

運用ログ / debug log / trace session manifest / trace shaping。

### `src/core/records/`

episode record、unknown overlay、improvement candidate など、保存系の圧縮レコード。

---

## 5. `src/llm/` の責務

外部 evaluator / teacher / provider adapter 群です。

原則:

- 補助輪であり中核ではない
- 辞書・relation ロジックを置かない
- unavailable でもパイプラインが止まらないようにする
- LLM に渡す自然言語プロンプト本文は `settings/teacher_profiles.yaml` 側で管理し、コードへ埋め込まない

---

## 6. `tools/` の責務

## 6.1 統一入口

### `tools/lexicon_cli.py`

辞書メンテナンス用の共通入口です。

扱うサブコマンド:

- `convert-to-binary`
- `convert-from-binary`
- `profile-load`
- `augment-conversation`
- `bootstrap-ja`

## 6.2 個別ツール

### `tools/convert_dict_to_binary.py`

辞書変換本体。

### `tools/convert_binary_to_dict.py`

辞書逆変換本体。

### `tools/profile_lexicon_load.py`

ロード経路のプロファイル本体。

### `tools/augment_conversation_lexicon.py`

会話 seed マージ専用。

### `tools/bootstrap_japanese_lexicon.py`

大規模な日本語辞書構築専用。

方針:

- 入口はまとめる
- 重い実装本体は無理に合体させない

---

## 7. 設定と生成物の置き場

## 7.1 `settings/`

- `LLM_order.yaml`
- `scoring_v1.yaml`
- `teacher_profiles.yaml`

実験条件や外部順序の置き場です。
特に `teacher_profiles.yaml` は、external evaluator / teacher / input generator / lexicon enricher に渡す **LLM 向けプロンプト本文と payload defaults の正式な置き場** とします。

## 7.2 `runtime/`

生成物専用です。

主なサブディレクトリ:

- `runtime/logs/`
- `runtime/traces/`
- `runtime/episodes/`
- `runtime/learning_runs/`
- `runtime/dictionaries/`
- `runtime/cache/`

これらはソースではなく、**実行結果・補助辞書・cache** を置く層です。

---

## 8. 境界の結論

v4 のアーキテクチャで崩してはいけない線は次の 4 本です。

1. `Dictionary` と `Runtime` を混ぜない
2. `Trace` と `Episode` を混ぜない
3. `App entrypoint` と `Core logic` を混ぜない
4. `Tool entrypoint` と `Heavy implementation` を混ぜない

---

## 9. 今回の整理で固定したこと

- `main.py` をアプリの共通入口として固定
- `tools/lexicon_cli.py` を辞書メンテの共通入口として追加
- `src/apps/cli_common.py` で CLI 引数を共通化
- `runtime/` を生成物置き場として再定義
- 過去の runtime 生成物はリポジトリ本体から外す方針を固定

---

## 10. 結論

現時点の v4 は、

- **アプリ入口**
- **辞書メンテ入口**
- **core ロジック**
- **settings**
- **runtime 生成物**

の線を明確にした構造へ整理するのが最も壊れにくいです。
