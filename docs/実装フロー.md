# LSLM v3 実装フロー（最適版）

本ドキュメントは、LSLM v3 の思想・現行アーキテクチャ・実装安全性の3観点から最適化された実装フローを定義する。

---

## 0. 前提

LSLM v3 は以下の3分離を厳守する。

* 辞書 = 知識（Axis / Grammar / Slot / Relation）
* 実行時 = 状態（State / Policy）
* 保存 = 記録（Log / Training Data / Evaluation）

この分離を崩さないことが最重要原則である。

---

## 1. スキーマ固定

### 目的

全モジュールの入出力契約を先に確定し、後工程での破綻を防ぐ。

### 定義対象

* dict.json エントリ構造
* 実行時 State 構造
* IntentPlan
* RecallCandidate
* SlotFrame / FilledSlots
* SurfacePlan
* ResponseResult
* ログ JSONL フォーマット

### 完了条件

* 全構造が dataclass または JSON Schema で定義済み
* 各モジュール間のデータ受け渡しが明文化されている

---

## 2. 辞書仕様の最小完成

### 目的

LSLM v3 の知識構造を最小単位で成立させる。

### 必須要素

* Axis（10軸）
* Grammar
* Slot（基本フレーム）
* Relation

### 完了条件

* 少数語彙で dict.json が成立
* すべてのエントリが上記要素を持つ

---

## 3. 辞書 I/O 基盤

### 目的

辞書を安定してロード・変換・参照できるようにする。

### 実装内容

* dict.json → .lsdx 変換
* .lsdx ロード
* インデックス生成
* バリデーション

### 完了条件

* load_lexicon_container が安定動作
* 主要フィールド（Axis / Grammar / Slot / Relation）取得可能

---

## 4. ログ基盤の常設

### 目的

全思考プロセスを可視化し、後の学習・分析を可能にする。

### 記録対象

* input
* intent
* recall候補
* slot構造
* 候補文
* スコア内訳
* 棄却理由
* relation経路
* 最終出力

### 完了条件

* 1入力につき1ログが JSONL で出力される
* 中間状態がすべて追跡可能

---

## 5. 最小縦スライス実装

### 目的

システムを1本通して動作させる

### 制約

* intent は固定（1〜2種類）
* 高度な分岐は禁止

### フロー

input
→ semantic recall
→ slot filling
→ surface realization
→ scoring
→ response

### 完了条件

* 任意入力で必ず1応答が返る
* ログが完全に出力される

---

## 6. Semantic Recall v1

### 目的

入力から意味的候補を発散する

### 手法

* 入力語
* relation展開
* axis近傍探索

### 完了条件

* 複数候補が安定して生成される

---

## 7. Grammar 前段フィルタ

### 目的

不正な候補を早期除去する

### 判定

* 品詞
* 接続可否
* function/content

### 完了条件

* 明らかな不正候補が除去される

---

## 8. Slot Filling v1

### 目的

意味構造を構築する

### 対象

* actor
* target
* state
* time
* location

### 完了条件

* slot が埋まる
* 未充填 slot が明示される

---

## 9. Surface Realization v1

### 目的

意味構造を自然文に変換する

### 手法

* テンプレートベース

### 完了条件

* 文として成立する出力が生成される

---

## 10. Grammar 後段チェック + Scoring

### 目的

最終候補の品質評価

### スコア要素

* 意味整合
* slot充足率
* grammar適合率
* 入力保持率

### 完了条件

* 候補から1つ選択できる

---

## 11. Intent Planning 一般化

### 目的

応答タイプを拡張する

### 追加対象

* 質問
* 共感
* 確認
* 説明

### 完了条件

* 複数 intent が安定動作

---

## 12. 探索と収束の強化

### 目的

応答品質の向上

### 強化内容

* 候補数増加
* relation深度拡張
* axis距離最適化
* クラスタリング

### 完了条件

* 候補の質が向上

---

## 13. 学習・評価導入

### 目的

モデルの改善

### 内容

* 外部評価器導入（Gemini / OpenAI / Local）
* 教師データ生成（intent / slots / target）

### 完了条件

* 評価スコアがログに追加される
* 学習ループが回る

---

## 14. 辞書への還元

### 目的

知識の強化

### 対象

* relation追加候補
* slotフレーム補強
* axis微調整

### 注意

* 自動反映禁止
* 必ず検証を挟む

### 完了条件

* 辞書が改善されるが破綻しない

---

## 15. dataclass 定義（実装基盤）

以下は、LSLM v3 の最小実装から拡張実装までを見据えた dataclass 定義案である。

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


IntentType = Literal[
    "respond",
    "empathy",
    "question",
    "confirm",
    "explain",
    "unknown",
]


PolicyType = Literal[
    "answer",
    "ask_back",
    "agree",
    "hold",
    "clarify",
]


SlotName = Literal[
    "actor",
    "target",
    "recipient",
    "location",
    "time",
    "cause",
    "state",
    "predicate",
    "topic",
    "manner",
]


@dataclass(slots=True)
class AxisVector:
    valence: float = 0.0
    arousal: float = 0.0
    abstractness: float = 0.0
    sociality: float = 0.0
    temporality: float = 0.0
    agency: float = 0.0
    causality: float = 0.0
    certainty: float = 0.0
    deixis: float = 0.0
    discourse_force: float = 0.0


@dataclass(slots=True)
class GrammarConstraints:
    pos: str
    sub_pos: str = ""
    can_start: bool = False
    can_end: bool = False
    independent: bool = True
    content_word: bool = True
    function_word: bool = False
    requires_prev: List[str] = field(default_factory=list)
    requires_next: List[str] = field(default_factory=list)
    forbid_prev: List[str] = field(default_factory=list)
    forbid_next: List[str] = field(default_factory=list)


@dataclass(slots=True)
class SlotConstraint:
    name: str
    required: bool = False
    allowed_pos: List[str] = field(default_factory=list)
    semantic_hint: List[str] = field(default_factory=list)
    note: str = ""


@dataclass(slots=True)
class RelationEdge:
    relation: str
    target: str
    weight: float = 1.0
    bidirectional: bool = False
    note: str = ""


@dataclass(slots=True)
class LexiconEntry:
    word: str
    category: str
    hierarchy: List[str] = field(default_factory=list)
    vector: AxisVector = field(default_factory=AxisVector)
    grammar: GrammarConstraints = field(
        default_factory=lambda: GrammarConstraints(pos="unknown")
    )
    slots: List[SlotConstraint] = field(default_factory=list)
    relations: List[RelationEdge] = field(default_factory=list)
    frequency: float = 0.0
    style_tags: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LexiconMeta:
    semantic_axes: List[str] = field(
        default_factory=lambda: [
            "valence",
            "arousal",
            "abstractness",
            "sociality",
            "temporality",
            "agency",
            "causality",
            "certainty",
            "deixis",
            "discourse_force",
        ]
    )
    entry_count: int = 0
    version: str = "v3"


@dataclass(slots=True)
class LexiconIndexes:
    by_pos: Dict[str, List[str]] = field(default_factory=dict)
    can_start: List[str] = field(default_factory=list)
    can_end: List[str] = field(default_factory=list)
    content_words: List[str] = field(default_factory=list)
    function_words: List[str] = field(default_factory=list)
    entry_path: Dict[str, List[str]] = field(default_factory=dict)


@dataclass(slots=True)
class LexiconContainer:
    meta: LexiconMeta = field(default_factory=LexiconMeta)
    entries: Dict[str, LexiconEntry] = field(default_factory=dict)
    indexes: LexiconIndexes = field(default_factory=LexiconIndexes)


@dataclass(slots=True)
class InputState:
    raw_text: str
    tokens: List[str]
    normalized_tokens: List[str] = field(default_factory=list)
    timestamp: str = ""
    session_id: str = ""
    turn_id: str = ""


@dataclass(slots=True)
class DialogueState:
    current_topic: str = ""
    last_subject: str = ""
    last_object: str = ""
    referents: Dict[str, str] = field(default_factory=dict)
    context_vector: AxisVector = field(default_factory=AxisVector)
    inferred_intent_history: List[str] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IntentPlan:
    intent: IntentType = "unknown"
    confidence: float = 0.0
    required_slots: List[str] = field(default_factory=list)
    optional_slots: List[str] = field(default_factory=list)
    response_policy_hint: PolicyType = "hold"
    note: str = ""


@dataclass(slots=True)
class RecallCandidate:
    word: str
    score: float
    source: Literal["input", "relation", "axis", "fallback"] = "fallback"
    relation_path: List[str] = field(default_factory=list)
    axis_distance: float = 0.0
    grammar_ok: bool = True
    note: str = ""


@dataclass(slots=True)
class RecallResult:
    seeds: List[str] = field(default_factory=list)
    candidates: List[RecallCandidate] = field(default_factory=list)


@dataclass(slots=True)
class SlotValue:
    slot_name: str
    value: str
    confidence: float = 0.0
    source_candidate: str = ""
    inferred: bool = False
    note: str = ""


@dataclass(slots=True)
class SlotFrame:
    predicate: str = ""
    predicate_type: str = ""
    constraints: List[SlotConstraint] = field(default_factory=list)


@dataclass(slots=True)
class FilledSlots:
    frame: SlotFrame = field(default_factory=SlotFrame)
    values: Dict[str, SlotValue] = field(default_factory=dict)
    missing_required: List[str] = field(default_factory=list)
    optional_unfilled: List[str] = field(default_factory=list)
    consistency_score: float = 0.0


@dataclass(slots=True)
class SurfacePlan:
    template_id: str = ""
    style: str = "neutral"
    politeness: str = "plain"
    sentence_count: int = 1
    order: List[str] = field(default_factory=list)
    auxiliaries: List[str] = field(default_factory=list)
    note: str = ""


@dataclass(slots=True)
class RealizationCandidate:
    text: str
    token_sequence: List[str] = field(default_factory=list)
    template_id: str = ""
    grammar_violations: List[str] = field(default_factory=list)
    slot_coverage: float = 0.0
    semantic_score: float = 0.0
    final_score: float = 0.0


@dataclass(slots=True)
class ScoreBreakdown:
    semantic_consistency: float = 0.0
    slot_fitness: float = 0.0
    grammar_fitness: float = 0.0
    input_retention: float = 0.0
    policy_fitness: float = 0.0
    total: float = 0.0
    reasons: List[str] = field(default_factory=list)


@dataclass(slots=True)
class ResponseResult:
    text: str
    intent: IntentType = "unknown"
    policy: PolicyType = "hold"
    chosen_candidate: Optional[RealizationCandidate] = None
    score: ScoreBreakdown = field(default_factory=ScoreBreakdown)
    used_relations: List[str] = field(default_factory=list)
    used_slots: Dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationResult:
    evaluator_name: str
    score: float
    label: str = ""
    feedback: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TraceLog:
    session_id: str
    turn_id: str
    input_state: InputState
    dialogue_state: DialogueState
    intent_plan: IntentPlan
    recall_result: RecallResult
    filled_slots: FilledSlots
    surface_plan: SurfacePlan
    candidates: List[RealizationCandidate] = field(default_factory=list)
    response: Optional[ResponseResult] = None
    evaluation: List[EvaluationResult] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)
```

### 設計メモ

* `LexiconEntry` は辞書知識の最小単位
* `DialogueState` は会話ごとに変動する実行時状態
* `IntentPlan` は planner の出力
* `RecallResult` は semantic recall の出力
* `FilledSlots` は convergence の中核
* `SurfacePlan` と `RealizationCandidate` は表層化責務を分離する
* `TraceLog` は全工程のスナップショットを保持する

---

## まとめ

最適フローの本質は以下である。

* 実行順と実装順を分離する
* スキーマとログを最初に固める
* 縦スライスで1本通す
* 後から横に広げる

これにより、LSLM v3 の思想・実装・学習のすべてを破綻なく統合できる。
