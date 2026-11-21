"""Chat with a model on any OpenAI-compatible server through anyserve, printing tokens as they arrive."""
from __future__ import annotations
import argparse
import json

from anyserve import AnyserveClient, ExecutionOptions


def main() -> None:
    parser = argparse.ArgumentParser(description="Send one chat prompt through anyserve.")
    parser.add_argument("--endpoint", help="default: ANYSERVE_ENDPOINT or http://127.0.0.1:50052")
    parser.add_argument("--model", required=True, help="model name; the interface is openai/chat/<model>")
    parser.add_argument("--prompt", default="Say hello in one sentence.")
    args = parser.parse_args()
    request = {"messages": [{"role": "user", "content": args.prompt}], "max_tokens": 256, "stream": True}
    options = ExecutionOptions(total_timeout_ms=300_000)
    with AnyserveClient(args.endpoint).invoke(f"openai/chat/{args.model}", json.dumps(request).encode(), options=options) as events:
        for event in events:
            if event["type"] == "output":
                data = event["data"][5:].strip()  # one "data: ..." line per frame
                if data != b"[DONE]":
                    for choice in json.loads(data).get("choices", []):
                        print(choice.get("delta", {}).get("content") or "", end="", flush=True)
            elif event["type"] == "finished":
                print()
                if event["state"] != "succeeded":
                    raise SystemExit(f"{event['error']['code']}: {event['error']['message']}")


if __name__ == "__main__":
    main()
