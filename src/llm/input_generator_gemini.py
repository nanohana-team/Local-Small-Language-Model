from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import requests
import yaml
from dotenv import load_dotenv


class GeminiInputGenerator:
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-lite",
        timeout_sec: int = 60,
        enabled: bool = True,
        random_seed: int | None = None,
        batch_size: int = 20,
        config_path: str | None = None,
    ) -> None:
        root_dir = Path(__file__).resolve().parents[2]
        env_path = root_dir / ".env"
        load_dotenv(env_path)

        self.root_dir = root_dir
        self.config_path = Path(config_path) if config_path else (root_dir / "settings" / "config.yaml")
        self.timeout_sec = timeout_sec
        self.enabled = enabled
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.rng = random.Random(random_seed)
        self.batch_size = max(1, int(batch_size))
        self._pool: List[List[str]] = []

        self.model_order = self._load_model_order(default_model_name=model_name)

        print(f"[GEMINI INIT] env_path={env_path}")
        print(f"[GEMINI INIT] config_path={self.config_path}")
        print(f"[GEMINI INIT] api_key_present={bool(self.api_key)}")
        if self.api_key:
            print(f"[GEMINI INIT] api_key_prefix={self.api_key[:8]}...")
        print(f"[GEMINI INIT] model_order={self.model_order}")

        self.allowed_punctuation = {"。", "、", "！", "？", "!", "?"}
        self.predicate_hints = {
            "する", "した", "し", "して", "します",
            "いる", "い", "ある", "なる", "なった",
            "思う", "思った", "考える", "考えた",
            "行く", "行った", "来る", "来た", "帰る", "帰った",
            "食べる", "食べた", "飲む", "飲んだ",
            "寝る", "寝た", "起きる", "起きた",
            "見る", "見た", "読む", "読んだ",
            "終わる", "終わった", "始まる", "始まった",
            "疲れた", "眠い", "痛い", "嬉しい", "悲しい",
            "楽しい", "欲しい", "寒い", "暑い", "大丈夫",
            "できる", "したい", "行きたい", "食べたい",
        }
        self.particle_hints = {
            "は", "が", "を", "に", "へ", "で", "と", "も", "から", "まで",
            "より", "の", "や", "ね", "よ", "か", "な", "って", "ので", "けど",
        }

        self.category_templates: Dict[str, Dict[str, List[str] | str]] = {
            "daily": {
                "desc": "日常の短い会話文",
                "examples": [
                    "今日は少し眠いです。",
                    "部屋の掃除をしたいです。",
                    "今はちょっと休みたいです。",
                ],
            },
            "emotion": {
                "desc": "感情や気分を表す文",
                "examples": [
                    "今日はなんだか嬉しいです。",
                    "少し不安な気持ちがあります。",
                    "その話を聞いて安心しました。",
                ],
            },
            "physical": {
                "desc": "体調や身体感覚を表す文",
                "examples": [
                    "少し頭が痛いです。",
                    "今日は体がだるいです。",
                    "お腹がすいています。",
                ],
            },
            "plan": {
                "desc": "予定や行動予定を表す文",
                "examples": [
                    "明日は買い物に行く予定です。",
                    "来週の予定を立てています。",
                    "あとで友達に連絡します。",
                ],
            },
            "past": {
                "desc": "過去の出来事を表す文",
                "examples": [
                    "昨日は早く寝ました。",
                    "さっき本を読み終えました。",
                    "今日は電車が少し遅れました。",
                ],
            },
            "question": {
                "desc": "疑問や確認を含む文",
                "examples": [
                    "今日は何を食べますか？",
                    "このあと時間はありますか？",
                    "その予定で大丈夫ですか？",
                ],
            },
            "request": {
                "desc": "依頼・提案・確認の文",
                "examples": [
                    "あとで手伝ってくれますか？",
                    "少し休んだほうがいいです。",
                    "先に連絡しておきますね。",
                ],
            },
            "reason": {
                "desc": "理由や接続を含む文",
                "examples": [
                    "雨が降っているので外に出ません。",
                    "少し疲れたけど作業を続けます。",
                    "時間があるから散歩に行きます。",
                ],
            },
        }

        self.curriculum_patterns: List[Dict[str, object]] = [
            {"level": 1, "count": 4, "token_range": "2-4", "must_include": "短い基本文。主題または述語のどちらかが明確。"},
            {"level": 2, "count": 5, "token_range": "4-7", "must_include": "助詞を1つ以上含む自然な短文。"},
            {"level": 3, "count": 6, "token_range": "6-10", "must_include": "修飾語や時制を含む文。"},
            {"level": 4, "count": 5, "token_range": "8-14", "must_include": "理由・確認・疑問・予定などを含む少し複雑な文。"},
        ]

        self.fallback_inputs: List[List[str]] = [
            ["今日", "は", "少し", "眠い", "です", "。"],
            ["お腹", "が", "すいた", "ので", "何か", "食べたい", "です", "。"],
            ["明日", "は", "買い物", "に", "行く", "予定", "です", "。"],
            ["今日は", "仕事", "が", "少し", "忙しい", "です", "。"],
            ["部屋", "の", "掃除", "を", "したい", "です", "。"],
            ["少し", "頭", "が", "痛い", "です", "。"],
            ["今", "は", "ちょっと", "休みたい", "です", "。"],
            ["友達", "に", "連絡", "して", "みます", "。"],
            ["昨日", "は", "早く", "寝た", "ので", "少し", "元気", "です", "。"],
            ["雨", "が", "降って", "いる", "ので", "外", "に", "出ません", "。"],
            ["今日は", "何", "を", "食べる", "か", "迷って", "います", "。"],
            ["この", "本", "は", "とても", "面白い", "です", "。"],
            ["少し", "不安", "だけど", "頑張りたい", "です", "。"],
            ["その", "予定", "で", "大丈夫", "です", "か", "？"],
            ["来週", "の", "予定", "を", "考えて", "います", "。"],
            ["今日は", "静か", "で", "落ち着く", "感じ", "です", "。"],
            ["あとで", "少し", "散歩", "に", "行きたい", "です", "。"],
            ["体", "が", "だるい", "ので", "無理", "しません", "。"],
            ["このあと", "時間", "が", "あれば", "本", "を", "読みます", "。"],
            ["少し", "疲れた", "から", "早め", "に", "休みます", "。"],
            ["今日は", "気分", "が", "いい", "です", "。"],
            ["何か", "手伝える", "こと", "は", "あります", "か", "？"],
            ["明後日", "の", "会議", "の", "準備", "が", "まだ", "終わって", "いません", "。"],
            ["新しい", "カフェ", "に", "行って", "みたい", "です", "。"],
            ["子供", "が", "熱", "を", "出した", "ので", "病院", "へ", "行きます", "。"],
            ["部屋", "を", "片づけたら", "少し", "気分", "が", "軽く", "なりました", "。"],
            ["来週", "の", "旅行", "の", "計画", "を", "立てて", "います", "。"],
            ["今日は", "ニュース", "で", "気になる", "記事", "が", "ありました", "。"],
            ["友達", "と", "久しぶり", "に", "会う", "約束", "が", "あります", "。"],
            ["晩御飯", "を", "家", "で", "作る", "か", "外", "で", "食べる", "か", "迷っています", "。"],
        ]

    def _load_model_order(self, default_model_name: str) -> List[str]:
        if not self.config_path.exists():
            print(f"[GEMINI CONFIG] config not found: {self.config_path}")
            return [default_model_name]

        try:
            with self.config_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[GEMINI CONFIG] failed to load config: {type(e).__name__}: {e}")
            return [default_model_name]

        raw = data.get("llm-api-order", None)
        if not isinstance(raw, list):
            print("[GEMINI CONFIG] llm-api-order not found or not a list")
            return [default_model_name]

        models: List[str] = []
        for x in raw:
            s = str(x).strip()
            if not s:
                continue
            if s.startswith("gemini"):
                models.append(s)

        if not models:
            models = [default_model_name]

        deduped: List[str] = []
        seen = set()
        for m in models:
            if m in seen:
                continue
            seen.add(m)
            deduped.append(m)

        return deduped

    def _build_curriculum_plan(self) -> List[Tuple[str, int, str, str]]:
        categories = list(self.category_templates.keys())
        self.rng.shuffle(categories)

        plan: List[Tuple[str, int, str, str]] = []
        cat_index = 0

        for rule in self.curriculum_patterns:
            level = int(rule["level"])
            count = int(rule["count"])
            token_range = str(rule["token_range"])
            must_include = str(rule["must_include"])

            for _ in range(count):
                category = categories[cat_index % len(categories)]
                cat_index += 1
                plan.append((category, level, token_range, must_include))

        if len(plan) > self.batch_size:
            plan = plan[:self.batch_size]
        elif len(plan) < self.batch_size:
            while len(plan) < self.batch_size:
                category = categories[cat_index % len(categories)]
                cat_index += 1
                plan.append((category, 2, "4-7", "助詞を1つ以上含む自然な短文。"))

        return plan

    def _build_prompt(self) -> str:
        plan = self._build_curriculum_plan()

        plan_lines: List[str] = []
        for idx, (category, level, token_range, must_include) in enumerate(plan, start=1):
            desc = str(self.category_templates[category]["desc"])
            plan_lines.append(
                f'{idx}. category="{category}" level={level} tokens={token_range} desc="{desc}" must="{must_include}"'
            )

        category_examples: List[str] = []
        for category, meta in self.category_templates.items():
            examples = meta["examples"]
            if isinstance(examples, list):
                ex = " / ".join(str(x) for x in examples[:2])
            else:
                ex = ""
            category_examples.append(f'- {category}: {meta["desc"]} 例: {ex}')

        prompt = f"""
あなたは日本語の学習用短文データ生成器です。
Local Small Language Model の学習用に、ちょうど {self.batch_size} 件の入力データを生成してください。

最重要目的:
- 人間に自然なだけでなく、学習に向いた「構造が見えやすい短文」を作る
- 主題・述語・助詞・時制・理由・疑問などの文法的手がかりを含める
- 各件は短文1つで完結させる
- ランダム性は持たせるが、意味の薄い単語羅列は禁止

カテゴリ説明:
{chr(10).join(category_examples)}

生成計画:
{chr(10).join(plan_lines)}

厳守条件:
- 出力は STRICT JSON ONLY
- 形式は {{"items":[{{"tokens":["...","..."]}}, ...]}}
- items の件数はちょうど {self.batch_size} 件
- 各 tokens は日本語の短文を適切に分けた配列にする
- 句読点として「。」「、」「？」「！」は必要なら tokens に含めてよい
- 多くの文で、助詞または述語のどちらか、できれば両方を含める
- 文法的に不自然な単語列は禁止
- 固有名詞まみれは禁止
- 記号だけの出力は禁止
- 同じ構文の繰り返しは禁止
- 同じ語を不自然に何度も繰り返さない
- 「名詞だけ並べた文」「意味不明な抽象語の羅列」は禁止
- 説明文・Markdown・コードブロックは禁止
- JSON以外の文字を一切出力しない

悪い例:
{{"items":[
  {{"tokens":["公園","美味しい","連絡する","。"]}},
  {{"tokens":["学校","早い","。"]}},
  {{"tokens":["今日","寒い","いる","。"]}}
]}}

良い例:
{{"items":[
  {{"tokens":["今日","は","少し","眠い","です","。"]}},
  {{"tokens":["雨","が","降って","いる","ので","外","に","出ません","。"]}},
  {{"tokens":["このあと","時間","は","あります","か","？"]}},
  {{"tokens":["明日","は","買い物","に","行く","予定","です","。"]}},
  {{"tokens":["少し","頭","が","痛い","ので","休みたい","です","。"]}}
]}}
""".strip()

        return prompt

    def _build_endpoint(self, model_name: str) -> str:
        return (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_name}:generateContent?key={self.api_key}"
        )

    def _build_body(self) -> dict:
        return {
            "contents": [
                {
                    "parts": [
                        {"text": self._build_prompt()}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 1.0,
                "topP": 0.9,
                "topK": 32,
                "maxOutputTokens": 3072,
                "responseMimeType": "application/json",
            },
        }

    def _request_json_text_once(self, model_name: str) -> str:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not set")

        endpoint = self._build_endpoint(model_name)
        body = self._build_body()

        print(f"[GEMINI REQUEST] model={model_name}")

        response = requests.post(
            endpoint,
            headers={"Content-Type": "application/json"},
            json=body,
            timeout=self.timeout_sec,
        )

        print(f"[GEMINI RESPONSE] model={model_name} status={response.status_code}")

        if response.status_code == 429:
            preview = response.text[:500] if response.text else ""
            raise RuntimeError(f"RATE_LIMIT_429: {preview}")

        if not response.ok:
            preview = response.text[:1000] if response.text else ""
            raise RuntimeError(
                f"Gemini HTTP error model={model_name} status={response.status_code} body={preview}"
            )

        data = response.json()

        if not isinstance(data, dict):
            raise RuntimeError(f"Gemini returned invalid JSON response for model={model_name}")

        candidates = data.get("candidates", [])
        if not candidates:
            preview = json.dumps(data, ensure_ascii=False)[:1000]
            raise RuntimeError(f"Gemini returned no candidates for model={model_name}: {preview}")

        first_candidate = candidates[0]
        if not isinstance(first_candidate, dict):
            raise RuntimeError(f"Gemini returned malformed candidate for model={model_name}")

        content = first_candidate.get("content", {})
        if not isinstance(content, dict):
            preview = json.dumps(first_candidate, ensure_ascii=False)[:1000]
            raise RuntimeError(f"Gemini candidate content is missing for model={model_name}: {preview}")

        parts = content.get("parts", [])
        if not isinstance(parts, list) or not parts:
            preview = json.dumps(first_candidate, ensure_ascii=False)[:1000]
            raise RuntimeError(f"Gemini returned no content parts for model={model_name}: {preview}")

        text = ""
        for part in parts:
            if isinstance(part, dict) and "text" in part:
                text += str(part["text"])

        text = text.strip()
        if not text:
            preview = json.dumps(first_candidate, ensure_ascii=False)[:1000]
            raise RuntimeError(f"Gemini returned empty text for model={model_name}: {preview}")

        return text

    def _request_json_text(self) -> str:
        last_error: Exception | None = None

        for idx, model_name in enumerate(self.model_order, start=1):
            try:
                print(f"[GEMINI TRY] {idx}/{len(self.model_order)} model={model_name}")
                return self._request_json_text_once(model_name)

            except Exception as e:
                last_error = e
                msg = str(e)

                if "RATE_LIMIT_429" in msg:
                    print(f"[GEMINI FALLBACK] 429 on model={model_name}")
                    continue

                print(f"[GEMINI ERROR] model={model_name} {type(e).__name__}: {e}")
                raise

        if last_error is not None:
            raise RuntimeError(f"All Gemini models failed with 429. last_error={last_error}")

        raise RuntimeError("No Gemini models available")

    def _normalize_punctuation(self, s: str) -> str:
        table = {
            ",": "、",
            ".": "。",
            "!": "！",
            "?": "？",
        }
        return table.get(s, s)

    def _clean_tokens(self, tokens: object) -> List[str]:
        if not isinstance(tokens, list):
            return []

        cleaned: List[str] = []
        prev = ""

        for x in tokens:
            s = str(x).strip()
            if not s:
                continue

            s = self._normalize_punctuation(s)

            if s in {"「", "」", "『", "』", "（", "）", "(", ")", "[", "]", "{", "}"}:
                continue

            if s in self.allowed_punctuation:
                if not cleaned:
                    continue
                if prev in self.allowed_punctuation:
                    continue
                cleaned.append(s)
                prev = s
                continue

            cleaned.append(s)
            prev = s

        while cleaned and cleaned[-1] == "、":
            cleaned.pop()

        return cleaned[:30]

    def _has_particle(self, tokens: List[str]) -> bool:
        return any(t in self.particle_hints for t in tokens)

    def _has_predicate_hint(self, tokens: List[str]) -> bool:
        for t in tokens:
            if t in self.predicate_hints:
                return True
            if t.endswith(("い", "たい", "した", "する", "したい", "ます", "ません", "です", "でした")):
                return True
        return False

    def _is_low_quality(self, tokens: List[str]) -> Tuple[bool, str]:
        if len(tokens) < 2:
            return True, "too_short"

        non_punct = [t for t in tokens if t not in self.allowed_punctuation]
        if len(non_punct) < 2:
            return True, "too_few_content_tokens"

        if len(non_punct) >= 3 and len(set(non_punct)) <= 1:
            return True, "repetitive"

        punct_count = sum(1 for t in tokens if t in self.allowed_punctuation)
        if punct_count >= max(3, len(tokens) // 2):
            return True, "too_much_punctuation"

        if not self._has_particle(tokens) and not self._has_predicate_hint(tokens):
            return True, "no_structure_hint"

        if len(non_punct) >= 4 and not self._has_predicate_hint(tokens):
            return True, "no_predicate"

        bad_patterns = [
            ("名詞っぽい語の羅列回避", len(non_punct) >= 4 and all(len(t) >= 2 for t in non_punct[:4]) and not self._has_particle(tokens)),
        ]
        for reason, matched in bad_patterns:
            if matched:
                return True, reason

        return False, "ok"

    def _parse_batch_response(self, text: str) -> List[List[str]]:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise RuntimeError("Gemini response is not a JSON object")

        items = parsed.get("items", [])
        if not isinstance(items, list):
            raise RuntimeError("items is not a list")

        results: List[List[str]] = []
        seen = set()
        rejected: Dict[str, int] = {}

        for item in items:
            if not isinstance(item, dict):
                rejected["item_not_dict"] = rejected.get("item_not_dict", 0) + 1
                continue

            tokens = self._clean_tokens(item.get("tokens", []))
            if not tokens:
                rejected["empty_after_clean"] = rejected.get("empty_after_clean", 0) + 1
                continue

            is_bad, reason = self._is_low_quality(tokens)
            if is_bad:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue

            key = tuple(tokens)
            if key in seen:
                rejected["duplicate"] = rejected.get("duplicate", 0) + 1
                continue

            seen.add(key)
            results.append(tokens)

        print(f"[GEMINI PARSE] accepted={len(results)} rejected={rejected}")

        if not results:
            raise RuntimeError("generated batch is empty")

        self.rng.shuffle(results)
        return results

    def _call_gemini_batch(self) -> List[List[str]]:
        text = self._request_json_text()
        return self._parse_batch_response(text)

    def _fill_pool(self) -> None:
        batch = self._call_gemini_batch()
        self._pool.extend(batch)
        print(f"[GEMINI POOL] pooled={len(self._pool)}")

    def _fallback_generate(self) -> List[str]:
        result = list(self.rng.choice(self.fallback_inputs))
        print(f"[FALLBACK INPUT] generated={result}")
        return result

    def generate(self) -> List[str]:
        if not self.enabled:
            print("[GEMINI INFO] generator disabled, using fallback")
            return self._fallback_generate()

        try:
            if not self._pool:
                self._fill_pool()

            if self._pool:
                result = list(self._pool.pop())
                print(f"[GEMINI OK] generated={result}")
                return result

            print("[FALLBACK] Gemini did not return usable inputs. Falling back to local inputs.")
            return self._fallback_generate()

        except Exception as e:
            print(f"[GEMINI ERROR] {type(e).__name__}: {e}")
            print("[FALLBACK] Falling back to local inputs.")
            return self._fallback_generate()