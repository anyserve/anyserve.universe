"""Serve a vLLM (or any OpenAI-compatible) chat model through anyserve.

The payload is an OpenAI chat completions request body. Without ``"stream"``
the result is the response body; with ``"stream": true`` every server-sent
event line becomes an output frame, so callers see tokens as they are
produced. vLLM keeps the model, KV cache and batching; anyserve adds placement
across servers, deadlines and failover.
"""
from __future__ import annotations
import argparse
import json
import urllib.error
import urllib.request

from anyserve import WorkerError, serve, worker


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward OpenAI chat completions to a vLLM server.")
    parser.add_argument("--endpoint", default="http://127.0.0.1:50052", help="anyserve control plane")
    parser.add_argument("--vllm", default="http://127.0.0.1:8000", help="vLLM OpenAI-compatible base URL")
    parser.add_argument("--model", required=True, help="model name as served by vLLM")
    parser.add_argument("--api-key", help="bearer token for vLLM, if it requires one")
    parser.add_argument("--interface", help="default: vllm/<model>")
    parser.add_argument("--concurrency", type=int, default=16, help="in-flight requests per worker")
    parser.add_argument("--worker-id")
    args = parser.parse_args()
    url = f"{args.vllm.rstrip('/')}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    def open_completion(body: dict):
        request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
        try:
            return urllib.request.urlopen(request, timeout=600)
        except urllib.error.HTTPError as error:
            # The request itself was rejected: another worker would reject it too.
            raise WorkerError(f"vllm returned HTTP {error.code}: {error.read().decode(errors='replace')[:200]}")
        except (urllib.error.URLError, TimeoutError) as error:
            raise WorkerError(f"vllm unreachable: {error}", retryable=True)

    @worker(interface=args.interface or f"vllm/{args.model}", attributes={"model": args.model},
            max_active_leases=args.concurrency)
    def chat(payload: bytes):
        body = json.loads(payload)
        body.setdefault("model", args.model)
        response = open_completion(body)
        if not body.get("stream"):
            with response:
                return response.read()
        return stream_events(response)

    serve(chat, args.endpoint, worker_id=args.worker_id)


def stream_events(response):
    """Yield each ``data:`` line of the event stream as one output frame."""
    with response:
        for line in response:
            if line.startswith(b"data:"):
                yield line.rstrip(b"\r\n") + b"\n"


if __name__ == "__main__":
    main()
