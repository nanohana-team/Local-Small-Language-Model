import argparse
import json
import socket
import uuid
from typing import Any, Dict, List


def send_message(host: str, port: int, message: Dict[str, Any]) -> str:
    payload = json.dumps(message, ensure_ascii=False).encode("utf-8")

    with socket.create_connection((host, port), timeout=10.0) as sock:
        sock.sendall(payload)
        try:
            response = sock.recv(4096)
            return response.decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""


class LoopStarter:
    def __init__(
        self,
        entry_host: str,
        entry_ports: List[int],
        ttl: int,
    ) -> None:
        self.entry_host = entry_host
        self.entry_ports = entry_ports
        self.ttl = ttl

    def build_message(self, text: str) -> Dict[str, Any]:
        return {
            "session_id": str(uuid.uuid4()),
            "input": text,
            "ttl": self.ttl,
            "max_ttl": self.ttl,
            "history": [],
        }

    def run_once(self, text: str) -> None:
        message = self.build_message(text)

        print("[START] loop message")
        print(json.dumps(message, ensure_ascii=False, indent=2))

        last_error = None

        for port in self.entry_ports:
            try:
                print(f"[SEND] -> {self.entry_host}:{port}")
                response = send_message(self.entry_host, port, message)
                if response:
                    print(f"[RECV] {response}")
                else:
                    print("[RECV] (no response)")
                return
            except Exception as exc:
                last_error = exc
                print(f"[WARN] failed to send to {self.entry_host}:{port}: {exc}")

        if last_error is not None:
            raise RuntimeError(f"all entry ports failed: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start a loop message and inject it into one of the LLM nodes."
    )
    parser.add_argument(
        "text",
        nargs="?",
        default="こんにちは。今日の話題を決めてください。",
        help="Initial input text",
    )
    parser.add_argument("--entry-host", default="127.0.0.1", help="Entry host")
    parser.add_argument(
        "--entry-ports",
        type=int,
        nargs="+",
        default=[3000, 3001, 3002],
        help="Candidate entry ports",
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=5,
        help="Number of loops (1 loop = 3 turns when using 3 nodes)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # 3ノードのリング想定なので、1 loop = 3 応答
    ttl = args.loops * 3

    starter = LoopStarter(
        entry_host=args.entry_host,
        entry_ports=args.entry_ports,
        ttl=ttl,
    )
    starter.run_once(args.text)


if __name__ == "__main__":
    main()
