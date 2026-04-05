from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import List

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

        self.fallback_inputs: List[List[str]] = [
            ["私", "は", "元気", "です"],
            ["今日は", "疲れた"],
            ["仕事", "終わった"],
            ["眠い"],
            ["お腹", "すいた"],
            ["楽しい"],
            ["悲しい"],
            ["ゲーム", "したい"],
            ["外", "寒い"],
            ["眠れない"],
            ["明日", "忙しい"],
            ["少し", "休みたい"],
            ["何か", "食べたい"],
            ["気分", "が", "いい"],
            ["ちょっと", "だるい"],
            ["散歩", "したい"],
            ["雨", "降ってる"],
            ["頭", "痛い"],
            ["やる気", "出ない"],
            ["今日は", "静か"],
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

    def _build_prompt(self) -> str:
        return f"""
あなたは日本語の短文入力データ生成器です。
Local Small Language Model の学習用に、自然な文章を {self.batch_size} 件生成してください。

条件:
- 出力は STRICT JSON ONLY
- 形式は {{"items":[{{"tokens":["...", "..."]}}, {{"tokens":["..."]}}]}}
- items の件数はちょうど {self.batch_size} 件
- 各 tokens は 1〜30 個
- 日本語
- 単語列として分かちやすい形にする
- 記号だけの出力は禁止
- 同じパターンに偏りすぎないよう、できるだけランダム性を持たせる
- 各件はできるだけ別パターンにする
- 説明文・Markdown・コードブロックは禁止
- JSON以外の文字を一切出力しない

良い例:
{{"items":[
  {{"tokens":["今日","は","眠い","です"]}},
  {{"tokens":["仕事","が","終わった","。"]}},
  {{"tokens":["今日は","少し","疲れた","かも","しれない","。"]}},
  {{"tokens":["お腹","すいた","、","何か","食べ","たい","。"]}},
  {{"tokens":["ゲーム","が","したい","な"]}}
]}}
""".strip()

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
                "temperature": 1.15,
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": 2048,
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

    def _clean_tokens(self, tokens: object) -> List[str]:
        if not isinstance(tokens, list):
            return []

        cleaned: List[str] = []
        for x in tokens:
            s = str(x).strip()
            if not s:
                continue
            if s in {"。", "、", ",", ".", "!", "！", "?", "？"}:
                continue
            cleaned.append(s)

        return cleaned[:30]

    def _parse_batch_response(self, text: str) -> List[List[str]]:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise RuntimeError("Gemini response is not a JSON object")

        items = parsed.get("items", [])
        if not isinstance(items, list):
            raise RuntimeError("items is not a list")

        results: List[List[str]] = []
        seen = set()

        for item in items:
            if not isinstance(item, dict):
                continue
            tokens = self._clean_tokens(item.get("tokens", []))
            if not tokens:
                continue

            key = tuple(tokens)
            if key in seen:
                continue
            seen.add(key)
            results.append(tokens)

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

    def generate(self) -> List[str]:
        if not self.enabled:
            print("[GEMINI INFO] generator disabled, using fallback")
            return list(self.rng.choice(self.fallback_inputs))

        try:
            if not self._pool:
                self._fill_pool()

            if self._pool:
                result = list(self._pool.pop())
                print(f"[GEMINI OK] generated={result}")
                return result

            print("[FALLBACK] Gemini did not return usable inputs. Falling back to local inputs.")
            return list(self.rng.choice(self.fallback_inputs))

        except Exception as e:
            print(f"[GEMINI ERROR] {type(e).__name__}: {e}")
            print("[FALLBACK] Falling back to local inputs.")
            return list(self.rng.choice(self.fallback_inputs))