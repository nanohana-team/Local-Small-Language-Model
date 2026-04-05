生成内容:
- src/apps/convert_dict_to_binary.py
- src/apps/repeat_divergence_convergence.py
- src/apps/learning_loop.py
- src/core/io/lsd_lexicon.py
- src/core/primitive/divergence.py
- src/core/primitive/convergence.py

使い方例:
1) JSON -> 高速バイナリ(.lsdx)へ変換
   python -m src.apps.convert_dict_to_binary dict.old2.json --format lsdx --verify

2) 既存スクリプトで利用
   python -m src.apps.repeat_divergence_convergence --lexicon dict.old2.lsdx --words 私 は 元気
   python -m src.apps.learning_loop --lexicon dict.old2.lsdx

備考:
- .lsdx は mmap 前提の indexed 形式
- .lsd は zlib 圧縮の単一バイナリ形式
- ロード / セーブ時に stderr へ進捗バーを出します
