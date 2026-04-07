# README.md

## Local Small Language Model v3.1.0a

LSLM は、軽量環境で動作する
**探索型言語エンジン**の実装プロジェクトです。

---

## ✨ 特徴

* 小規模・CPU前提
* 発散と収束による思考
* 辞書ベースの意味構造
* 完全ログ可視化
* 外部LLMによる評価連携

---

## 🧠 コンセプト

LSLM は「LLMの縮小版」ではありません。

代わりに、

> 思考プロセスそのものを再現する

ことを目的としています。

---

## 📦 現在の構成

```
Local-Small-Language-Model/
├─ libs/
│  └─ dict.lsdx
├─ src/
│  └─ core/io/lsd_lexicon.py
├─ tools/
│  ├─ convert_dict_to_binary.py
│  └─ dict.json
├─ settings/
│  └─ config.yaml
```

---

## 🚀 できること（v3.1.0a）

* 辞書JSONのロード
* バイナリ辞書への変換
* 高速辞書アクセス（.lsdx）
* 階層構造の自動生成

---

## ⚠️ まだできないこと

* 会話生成
* 推論エンジン
* 学習ループ

これらは今後の実装対象です。

---

## 🔧 使用例

```bash
python -m tools.convert_dict_to_binary tools/dict.json -o libs/dict.lsdx --format lsdx --verify
```

---

## 🧩 設計のキーポイント

### 辞書 = 知識

* Axis
* Grammar
* Slot
* Relation

### 実行時 = 状態

* State
* Policy

### 保存 = 記録

* Log
* Training Data
* Evaluation

---

## 🛣️ 今後のロードマップ

1. 推論パイプライン実装
2. スロット充填強化
3. スコアリング導入
4. 強化学習
5. 自己改善ループ

---

## 🎯 ゴール

* 軽量で高速
* 解釈可能
* 学習可能

そんな

> 「考えることができる小さな言語モデル」

を実現することです。

---

## 📄 ライセンス

LICENSE参照