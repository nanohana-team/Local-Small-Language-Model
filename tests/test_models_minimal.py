# test_models_dotenv.py
import os
from dotenv import load_dotenv

load_dotenv()  # ←これで.env読む

def test_openai():
    try:
        from openai import OpenAI
    except Exception as e:
        print(f"[OPENAI] import failed: {e}")
        return False

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        resp = client.responses.create(
            model="gpt-5.4",
            input="Reply with exactly: OK"
        )
        print("[OPENAI] success:", resp.output_text)
        return True
    except Exception as e:
        print("[OPENAI] failed:", e)
        return False


def test_gemini():
    try:
        from google import genai
    except Exception as e:
        print(f"[GEMINI] import failed: {e}")
        return False

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    try:
        resp = client.models.generate_content(
            model="gemini-3.1-pro-preview",  # ←ここは環境依存
            contents="Reply with exactly: OK"
        )
        print("[GEMINI] success:", resp.text)
        return True
    except Exception as e:
        print("[GEMINI] failed:", e)
        return False


if __name__ == "__main__":
    ok1 = test_openai()
    ok2 = test_gemini()

    print("\n=== RESULT ===")
    print("gpt-5.4:", "OK" if ok1 else "NG")
    print("gemini-4.1-pro:", "OK" if ok2 else "NG")