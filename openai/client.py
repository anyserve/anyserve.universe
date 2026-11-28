"""Call an OpenAI-compatible model through anyserve (chat, completions, or embeddings)."""
from __future__ import annotations

import argparse
import json
import sys
import time

from anyserve import AnyserveClient, ExecutionOptions


def parse_attrs(items: list[str]) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"invalid attribute {item!r}: expected KEY=VALUE")
        key, value = item.split("=", 1)
        attrs[key] = value
    return attrs


def main() -> None:
    parser = argparse.ArgumentParser(description="Call an OpenAI-compatible model through anyserve.")
    parser.add_argument("--endpoint", help="default: ANYSERVE_ENDPOINT or http://127.0.0.1:50052")
    parser.add_argument("--model", required=True, help="model name; the interface is openai/<mode>/<model>")
    parser.add_argument("--mode", choices=("chat", "completions", "embeddings"), default="chat")
    parser.add_argument("--prompt", default="Say hello in one sentence.")
    parser.add_argument("--system", help="optional system message (chat mode)")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--no-stream", action="store_true", help="wait for the full JSON reply instead of streaming")
    parser.add_argument("--timeout-ms", type=int, default=300_000)
    parser.add_argument("--attr", action="append", default=[], metavar="KEY=VALUE", help="preferred worker attribute (repeatable)")
    parser.add_argument("--require", action="append", default=[], metavar="KEY=VALUE", help="required worker attribute (repeatable)")
    parser.add_argument("--stats", action="store_true", help="print timing metrics to stderr")
    parser.add_argument("--json", action="store_true", help="print the raw JSON response (non-streaming)")
    args = parser.parse_args()

    options = ExecutionOptions(
        total_timeout_ms=args.timeout_ms,
        preferred_attributes=parse_attrs(args.attr),
        required_attributes=parse_attrs(args.require),
    )
    client = AnyserveClient(args.endpoint)
    interface = f"openai/{args.mode}/{args.model}"
    started = time.perf_counter()

    if args.mode == "embeddings":
        body = {"input": [args.prompt]}
        raw = client.call(interface, json.dumps(body).encode(), options=options)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if args.json:
            print(raw.decode())
        else:
            reply = json.loads(raw)
            vec = reply["data"][0]["embedding"]
            preview = ", ".join(f"{x:.4f}" for x in vec[:6])
            print(f"embedding dim={len(vec)} [{preview}, ...]")
        if args.stats:
            print(f"elapsed={elapsed_ms:.1f}ms", file=sys.stderr)
        return

    if args.mode == "chat":
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.prompt})
        request: dict = {"messages": messages, "max_tokens": args.max_tokens}
    else:
        request = {"prompt": args.prompt, "max_tokens": args.max_tokens}
    if args.temperature is not None:
        request["temperature"] = args.temperature

    if args.no_stream or args.json:
        request["stream"] = False
        raw = client.call(interface, json.dumps(request).encode(), options=options)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if args.json:
            print(raw.decode())
        else:
            reply = json.loads(raw)
            choice = reply["choices"][0]
            text = choice["message"]["content"] if args.mode == "chat" else choice["text"]
            print(text)
        if args.stats:
            print(f"elapsed={elapsed_ms:.1f}ms", file=sys.stderr)
        return

    request["stream"] = True
    first_token_ms: float | None = None
    chunks = 0
    with client.invoke(interface, json.dumps(request).encode(), options=options) as events:
        for event in events:
            if event["type"] == "output":
                data = event["data"][5:].strip()  # one "data: ..." line per frame
                if data == b"[DONE]":
                    continue
                for choice in json.loads(data).get("choices", []):
                    piece = (
                        choice.get("delta", {}).get("content")
                        if args.mode == "chat"
                        else choice.get("text")
                    )
                    if piece:
                        if first_token_ms is None:
                            first_token_ms = (time.perf_counter() - started) * 1000
                        chunks += 1
                        print(piece, end="", flush=True)
            elif event["type"] == "finished":
                print(flush=True)
                if event["state"] != "succeeded":
                    err = event.get("error") or {}
                    raise SystemExit(f"{err.get('code', event['state'])}: {err.get('message', 'failed')}")
    if args.stats:
        total_ms = (time.perf_counter() - started) * 1000
        ttft = f"{first_token_ms:.1f}ms" if first_token_ms is not None else "n/a"
        print(f"ttft={ttft} total={total_ms:.1f}ms chunks={chunks}", file=sys.stderr)


if __name__ == "__main__":
    main()
