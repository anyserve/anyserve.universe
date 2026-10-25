"""Chat with a vLLM model through anyserve, streaming tokens as they arrive."""
from __future__ import annotations
import argparse
import json

from anyserve import AnyserveClient, ExecutionOptions


def main() -> None:
    parser = argparse.ArgumentParser(description="Send one chat prompt through anyserve.")
    parser.add_argument("--endpoint", default="http://127.0.0.1:50052")
    parser.add_argument("--model", required=True, help="model name; the interface is vllm/<model>")
    parser.add_argument("--prompt", default="Say hello in one sentence.")
    parser.add_argument("--no-stream", action="store_true", help="wait for the whole reply instead of streaming")
    args = parser.parse_args()
    client = AnyserveClient(args.endpoint)
    interface = f"vllm/{args.model}"
    request = {"messages": [{"role": "user", "content": args.prompt}], "max_tokens": 256, "stream": not args.no_stream}
    options = ExecutionOptions(total_timeout_ms=300_000)
    if args.no_stream:
        reply = json.loads(client.call(interface, json.dumps(request).encode(), options=options))
        print(reply["choices"][0]["message"]["content"])
        return
    buffer = b""
    with client.invoke(interface, json.dumps(request).encode(), options=options) as events:
        for event in events:
            if event["type"] == "output":
                buffer += event["data"]
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    print_delta(line)
            elif event["type"] == "finished":
                print()
                if event["state"] != "succeeded":
                    raise SystemExit(f"{event['error']['code']}: {event['error']['message']}")


def print_delta(line: bytes) -> None:
    if not line.startswith(b"data:"):
        return
    data = line[5:].strip()
    if data == b"[DONE]":
        return
    for choice in json.loads(data).get("choices", []):
        print(choice.get("delta", {}).get("content", ""), end="", flush=True)


if __name__ == "__main__":
    main()
