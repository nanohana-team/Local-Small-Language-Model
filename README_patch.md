# unknown word patch

## 変更点
- 最長一致ベースを維持したまま、未一致部分を `unknown span` として塊で保持
- `InputState` に `tokenization` / `unknown_spans` を追加
- unknown span を LLM に問い合わせて overlay 辞書へ反映
- 反映後は同一ターン内で再トークナイズ
- `tools/merge_lexicon_overlay.py` で overlay を `dict.json` へマージ可能

## 主な追加ファイル
- `src/training/unknown_word_learner.py`
- `tools/merge_lexicon_overlay.py`

## 設定
`settings/pipeline.yaml`
- `pipeline.unknown_word.enabled`
- `pipeline.unknown_word.min_span_length`
- `pipeline.unknown_word.max_spans_per_turn`
- `pipeline.unknown_word.promote_threshold`
- `pipeline.unknown_word.pending_path`
- `pipeline.unknown_word.overlay_path`

## 依存関係
追加依存はありません。`requirements.txt` は現状維持です。

## マージ後の流れ
1. 実行して `runtime/lexicon_overlay.json` を育てる
2. `python -m tools.merge_lexicon_overlay libs/dict.json --overlay runtime/lexicon_overlay.json -o libs/dict.json`
3. `python -m tools.convert_dict_to_binary libs/dict.json -o libs/dict.lsdx --format lsdx --verify`
