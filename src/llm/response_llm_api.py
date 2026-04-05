import os
import yaml
import argparse
from typing import Optional

# dotenv
try:
    from dotenv import load_dotenv
    _dotenv_available = True
except Exception:
    _dotenv_available = False

# OpenAI
try:
    from openai import OpenAI
    _openai_available = True
except Exception:
    _openai_available = False

# Gemini
try:
    import google.generativeai as genai
    _gemini_available = True
except Exception:
    _gemini_available = False


# =========================
# .env Load
# =========================

def _load_env():
    if _dotenv_available:
        load_dotenv()
    else:
        print("[WARN] python-dotenv not installed (.env not loaded)")


# =========================
# Config Loader
# =========================

def _load_config(path: str = "settings/config.yaml") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =========================
# Error Helpers
# =========================

def _is_gemini_rate_limit_error(exc: Exception) -> bool:
    """
    Geminiの429系エラーをざっくり判定する
    """
    text = str(exc).lower()

    rate_limit_markers = [
        "429",
        "resource_exhausted",
        "quota",
        "rate limit",
        "too many requests",
    ]

    return any(marker in text for marker in rate_limit_markers)


def _is_openai_model_name(model_name: str) -> bool:
    name = model_name.lower()
    return (
        name.startswith("gpt")
        or name.startswith("o1")
        or name.startswith("o3")
        or name.startswith("o4")
    )


def _is_gemini_model_name(model_name: str) -> bool:
    return "gemini" in model_name.lower()


# =========================
# OpenAI
# =========================

def _call_openai(prompt: str, model: str) -> str:
    if not _openai_available:
        raise RuntimeError("OpenAI SDK not available")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)

    res = client.responses.create(
        model=model,
        input=prompt,
    )

    return res.output_text.strip()


# =========================
# Gemini
# =========================

def _call_gemini(prompt: str, model: str) -> str:
    if not _gemini_available:
        raise RuntimeError("Gemini SDK not available")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)
    m = genai.GenerativeModel(model)

    res = m.generate_content(prompt)

    if not hasattr(res, "text") or not res.text:
        raise RuntimeError(f"Gemini returned empty response: model={model}")

    return res.text.strip()


# =========================
# Main API
# =========================

def generate_response(prompt: str) -> str:
    """
    入力文字列をそのままLLMに投げて、
    config.yamlの順序でフォールバックする
    Geminiで429が返った場合は次モデルへ進む
    """

    _load_env()
    config = _load_config()

    order = config.get("llm-api-order", [])
    if not order:
        raise RuntimeError("llm-api-order is empty in config.yaml")

    last_error: Optional[Exception] = None

    for model_name in order:
        try:
            if _is_openai_model_name(model_name):
                print(f"[LLM TRY] OpenAI: {model_name}")
                response = _call_openai(prompt, model_name)
                print("\n==============================")
                print("[LLM REQUEST]")
                print(prompt)
                print("[LLM RESPONSE]")
                print(response)
                print("==============================\n")
                return response

            if _is_gemini_model_name(model_name):
                print(f"[LLM TRY] Gemini: {model_name}")
                response = _call_gemini(prompt, model_name)
                print("\n==============================")
                print("[LLM REQUEST]")
                print(prompt)
                print("[LLM RESPONSE]")
                print(response)
                print("==============================\n")
                return response
            raise RuntimeError(f"Unknown model type: {model_name}")

        except Exception as e:
            last_error = e

            if _is_gemini_model_name(model_name) and _is_gemini_rate_limit_error(e):
                print(f"[LLM FALLBACK] Gemini rate limit (429) on {model_name} -> trying next model")
                continue

            print(f"[LLM FALLBACK] {model_name} failed: {e}")
            continue

    raise RuntimeError(f"All LLM API calls failed. Last error: {last_error}")


# =========================
# CLI Entry
# =========================

def _main():
    parser = argparse.ArgumentParser(description="LLM Response API CLI")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input text to send to LLM"
    )

    args = parser.parse_args()

    try:
        result = generate_response(args.input)
        print(result)
    except Exception as e:
        print(f"[ERROR] {e}")


if __name__ == "__main__":
    _main()