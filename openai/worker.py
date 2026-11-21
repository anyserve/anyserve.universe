"""Put any OpenAI-compatible inference server behind anyserve.

vLLM, Ollama, LM Studio, llama.cpp's server, SGLang and embedding servers all
speak the same HTTP API, so one connector covers them. It asks the server for
its models and registers one interface per model and endpoint:

    openai/chat/<model>          POST <base>/chat/completions
    openai/completions/<model>   POST <base>/completions
    openai/embeddings/<model>    POST <base>/embeddings

The payload is the request body as the OpenAI API takes it. The result is the
response body, or one output frame per server-sent event when the request asks
to stream.
"""
from __future__ import annotations
import argparse
import json
import os
import time
import urllib.error
import urllib.request

from anyserve import WorkerError, serve, worker

PATHS = {"chat": "/chat/completions", "completions": "/completions", "embeddings": "/embeddings"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve an OpenAI-compatible server through anyserve.")
    parser.add_argument("--endpoint", help="anyserve control plane; default: ANYSERVE_ENDPOINT or http://127.0.0.1:50052")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="the server's OpenAI base URL")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"), help="bearer token for the server; default: OPENAI_API_KEY")
    parser.add_argument("--model", action="append", help="model to serve (repeatable); default: every model the server lists")
    parser.add_argument("--serve", action="append", choices=PATHS, help="endpoint to register (repeatable); default: chat")
    parser.add_argument("--concurrency", type=int, default=16, help="requests in flight on this server, across its interfaces")
    parser.add_argument("--attr", action="append", default=[], metavar="KEY=VALUE", help="worker attribute for placement, such as site=edge (repeatable)")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    models = args.model or list_models(base, headers)
    endpoints = args.serve or ["chat"]
    attributes = dict(item.partition("=")[::2] for item in args.attr)
    slots = max(1, args.concurrency // (len(models) * len(endpoints)))
    handlers = [forwarder(base + PATHS[name], headers, model, f"openai/{name}/{model}", attributes, slots)
                for model in models for name in endpoints]
    print(f"serving {base} as " + ", ".join(handler.spec.interface for handler in handlers), flush=True)
    serve(handlers, endpoint=args.endpoint)


def list_models(base: str, headers: dict[str, str]) -> list[str]:
    """Return the ids the server lists, waiting while it is still loading."""
    while True:
        try:
            with urllib.request.urlopen(urllib.request.Request(f"{base}/models", headers=headers), timeout=10) as response:
                models = [entry["id"] for entry in json.load(response)["data"]]
            if not models:
                raise SystemExit(f"{base}/models lists no model; name one with --model")
            return models
        except urllib.error.HTTPError as error:
            raise SystemExit(f"{base}/models returned HTTP {error.code}")
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(2)


def forwarder(url: str, headers: dict[str, str], model: str, interface: str, attributes: dict[str, str], slots: int):
    @worker(interface=interface, attributes=attributes, max_active_leases=slots)
    def forward(payload: bytes):
        try:
            body = json.loads(payload)
            body.setdefault("model", model)
        except (ValueError, AttributeError):
            raise WorkerError("the payload must be a JSON object: an OpenAI request body")
        request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
        try:
            response = urllib.request.urlopen(request, timeout=600)
        except urllib.error.HTTPError as error:
            detail = f"HTTP {error.code}: {error.read().decode(errors='replace')[:300]}"
            # Overload and server faults may pass on another server; a rejected request will not.
            raise WorkerError(detail, retryable=error.code == 429 or error.code >= 500)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            raise WorkerError(f"server unreachable: {error}", retryable=True)
        if body.get("stream"):
            return events(response)
        with response:
            return response.read()

    return forward


def events(response):
    """Yield each ``data:`` line of the event stream as one output frame."""
    with response:
        for line in response:
            if line.startswith(b"data:"):
                yield line


if __name__ == "__main__":
    main()
