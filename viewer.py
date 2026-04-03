import time
from pathlib import Path
import os

log_path = Path("runtime/live_view.txt")

last_text = ""

while True:
    if log_path.exists():
        text = log_path.read_text(encoding="utf-8")
        if text != last_text:
            os.system("cls")  # Windows用クリア
            print(text)
            last_text = text
    time.sleep(0.2)