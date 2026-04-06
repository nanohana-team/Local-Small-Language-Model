# settings/ の使い方

## config.yaml
- パス設定
- `llm-api-order` の優先順

## pipeline.yaml
- intent / recall / slot / surface / scorer の数値

## learning.yaml
- learn / auto-learn の既定値
- reward / policy memory / teacher guidance の数値
- target / evaluator / input generator の LLM 数値

## 反映優先順位
1. コマンドライン引数
2. settings/*.yaml
3. src/utils/settings.py の内部デフォルト
