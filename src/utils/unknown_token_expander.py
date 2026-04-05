from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

try:
    import google.generativeai as genai
except Exception:
    genai = None


class UnknownTokenExpander:
    def __init__(
        self,
        dict_path: Path,
        model_name: str = "gemini-2.5-flash-lite",
        enabled: bool = True,
        backup: bool = True,
        min_token_len: int = 1,
        rate_limit_sec: float = 0.5,
    ) -> None:
        self.dict_path = Path(dict_path)
        self.model_name = model_name
        self.enabled = enabled
        self.backup = backup
        self.min_token_len = min_token_len
        self.rate_limit_sec = rate_limit_sec

        self._last_call_ts: float = 0.0
        self._known_tokens_cache: Optional[Set[str]] = None

        if self.enabled and genai is not None:
            api_key = os.getenv("GEMINI_API_KEY", "").strip()
            if api_key:
                genai.configure(api_key=api_key)
            else:
                print("[UNKNOWN] GEMINI_API_KEY not found; unknown token expansion disabled")
                self.enabled = False

        if self.enabled and genai is None:
            print("[UNKNOWN] google.generativeai is not available; unknown token expansion disabled")
            self.enabled = False

    def expand_from_episodes(self, episodes: List[Dict[str, Any]]) -> int:
        if not self.enabled:
            return 0

        unknowns = self.collect_unknown_tokens_from_episodes(episodes)
        if not unknowns:
            print("[UNKNOWN] no unknown tokens found")
            return 0

        print(f"[UNKNOWN] found unknown tokens: {unknowns}")

        added = 0
        for token in unknowns:
            try:
                if self.add_token_via_gemini(token):
                    added += 1
            except Exception as e:
                print(f"[UNKNOWN] failed to add token={token!r}: {e}")

        if added > 0:
            self._known_tokens_cache = None
            print(f"[UNKNOWN] added {added} tokens to dict")
        else:
            print("[UNKNOWN] no tokens added")

        return added

    def collect_unknown_tokens_from_episodes(self, episodes: List[Dict[str, Any]]) -> List[str]:
        known = self.load_known_tokens()
        found: List[str] = []
        seen: Set[str] = set()

        for ep in episodes:
            if not isinstance(ep, dict):
                continue

            for key in (
                "input_tokens",
                "output_tokens",
                "seed_tokens",
                "final_tokens",
                "converged_tokens",
            ):
                tokens = ep.get(key)
                if not isinstance(tokens, list):
                    continue

                for token in tokens:
                    if not isinstance(token, str):
                        continue
                    token = token.strip()
                    if len(token) < self.min_token_len:
                        continue
                    if token in known:
                        continue
                    if token in seen:
                        continue
                    if self._should_skip_token(token):
                        continue
                    seen.add(token)
                    found.append(token)

        return found

    def load_known_tokens(self) -> Set[str]:
        if self._known_tokens_cache is not None:
            return self._known_tokens_cache

        data = self._load_dict_json()
        known: Set[str] = set()

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    token = item.get("token") or item.get("surface") or item.get("word")
                    if isinstance(token, str) and token.strip():
                        known.add(token.strip())

        elif isinstance(data, dict):
            entries = data.get("entries")
            if isinstance(entries, list):
                for item in entries:
                    if isinstance(item, dict):
                        token = item.get("token") or item.get("surface") or item.get("word")
                        if isinstance(token, str) and token.strip():
                            known.add(token.strip())

        self._known_tokens_cache = known
        return known

    def add_token_via_gemini(self, token: str) -> bool:
        entry = self._generate_entry(token)
        if not entry:
            return False

        data = self._load_dict_json()

        if self.backup:
            self._backup_dict()

        if isinstance(data, list):
            data.append(entry)
        elif isinstance(data, dict):
            if "entries" not in data or not isinstance(data["entries"], list):
                data["entries"] = []
            data["entries"].append(entry)
        else:
            raise ValueError("Unsupported dict.json structure")

        self._save_dict_json(data)
        print(f"[UNKNOWN] added token={token!r}")
        return True

    def _generate_entry(self, token: str) -> Optional[Dict[str, Any]]:
        self._respect_rate_limit()

        model = genai.GenerativeModel(self.model_name)
        prompt = f"""
あなたは日本語の最小単位辞書を生成する補助器です。
次の未知語について、辞書に追加するためのJSON 1個だけを返してください。
説明文やコードブロックは禁止です。JSONオブジェクトのみ返してください。

未知語: {token}

要件:
- token: 元の語
- pos: 品詞（名詞 / 動詞 / 形容詞 / 形容動詞 / 副詞 / 接続詞 / 感動詞 / 連体詞 / 助詞 / 助動詞 のいずれか）
- reading: ひらがな。わからなければ元の語をそのまま
- base_form: 原形。わからなければ元の語をそのまま
- axes: 4次元の数値配列。各値は -1.0〜1.0
- notes: 短い説明

厳守:
- 必ずJSONオブジェクトのみ
- keysは token,pos,reading,base_form,axes,notes
- axesは要素数4
""".strip()

        response = model.generate_content(prompt)
        text = (getattr(response, "text", "") or "").strip()

        if not text:
            print(f"[UNKNOWN] empty Gemini response for token={token!r}")
            return None

        text = self._extract_json_object(text)
        obj = json.loads(text)

        if not isinstance(obj, dict):
            raise ValueError("Gemini response is not a JSON object")

        entry = self._normalize_entry(token, obj)
        return entry

    def _normalize_entry(self, token: str, obj: Dict[str, Any]) -> Dict[str, Any]:
        pos = str(obj.get("pos", "名詞")).strip() or "名詞"
        reading = str(obj.get("reading", token)).strip() or token
        base_form = str(obj.get("base_form", token)).strip() or token
        notes = str(obj.get("notes", "")).strip()

        axes = obj.get("axes", [0.0, 0.0, 0.0, 0.0])
        if not isinstance(axes, list) or len(axes) != 4:
            axes = [0.0, 0.0, 0.0, 0.0]

        norm_axes: List[float] = []
        for v in axes:
            try:
                fv = float(v)
            except Exception:
                fv = 0.0
            fv = max(-1.0, min(1.0, fv))
            norm_axes.append(fv)

        return {
            "token": token,
            "pos": pos,
            "reading": reading,
            "base_form": base_form,
            "axes": norm_axes,
            "notes": notes,
        }

    def _should_skip_token(self, token: str) -> bool:
        if not token:
            return True
        if token.isspace():
            return True
        return False

    def _load_dict_json(self) -> Any:
        if not self.dict_path.exists():
            raise FileNotFoundError(f"dict file not found: {self.dict_path}")

        with self.dict_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save_dict_json(self, data: Any) -> None:
        with self.dict_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _backup_dict(self) -> None:
        src = self.dict_path
        dst = src.with_suffix(src.suffix + ".bak")
        try:
            dst.write_bytes(src.read_bytes())
        except Exception as e:
            print(f"[UNKNOWN] backup failed: {e}")

    def _extract_json_object(self, text: str) -> str:
        text = text.strip()

        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Could not find JSON object in response: {text!r}")
        return text[start:end + 1]

    def _respect_rate_limit(self) -> None:
        now = time.time()
        delta = now - self._last_call_ts
        if delta < self.rate_limit_sec:
            time.sleep(self.rate_limit_sec - delta)
        self._last_call_ts = time.time()