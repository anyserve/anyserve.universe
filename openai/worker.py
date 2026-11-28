"""Put any OpenAI-compatible inference server behind anyserve.

vLLM, Ollama, LM Studio, llama.cpp's server, SGLang and embedding servers all
speak the same HTTP API, so one connector covers them. It asks the server for
its models and registers one interface per model and endpoint:

    openai/chat/<model>          POST <base>/chat/completions
    openai/completions/<model>   POST <base>/completions
    openai/embeddings/<model>    POST <base>/embeddings

The payload is the request body as the OpenAI API takes it. The result is the
response body, or one output frame per server-sent event when the request asks
to stream. Closing an Invoke cancels generation and closes the upstream stream.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

from anyserve import WorkerError, serve, worker

PATHS = {
    "chat": "/chat/completions",
    "completions": "/completions",
    "embeddings": "/embeddings",
}


def parse_attrs(items: list[str]) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"invalid --attr {item!r}: expected KEY=VALUE")
        key, value = item.split("=", 1)
        if not key:
            raise SystemExit(f"invalid --attr {item!r}: key must not be empty")
        attrs[key] = value
    return attrs


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve an OpenAI-compatible server through anyserve.")
    parser.add_argument("--endpoint", help="anyserve control plane; default: ANYSERVE_ENDPOINT or http://127.0.0.1:50052")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="the server's OpenAI base URL")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"), help="bearer token for the server; default: OPENAI_API_KEY")
    parser.add_argument("--model", action="append", help="model to serve (repeatable); default: every model the server lists")
    parser.add_argument("--serve", action="append", choices=PATHS, help="endpoint to register (repeatable); default: chat")
    parser.add_argument("--concurrency", type=int, default=16, help="requests in flight on this server, across its interfaces")
    parser.add_argument("--timeout", type=float, default=600.0, help="upstream HTTP timeout in seconds; default: 600")
    parser.add_argument("--pool", default="", help="worker pool name for tiering and spillover")
    parser.add_argument("--attr", action="append", default=[], metavar="KEY=VALUE", help="worker attribute for placement, such as site=edge (repeatable)")
    args = parser.parse_args()

    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")

    base = args.base_url.rstrip("/")
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    models = args.model or list_models(base, headers)
    endpoints = args.serve or ["chat"]
    attributes = parse_attrs(args.attr)
    slots = max(1, args.concurrency // (len(models) * len(endpoints)))
    handlers = [
        forwarder(
            base + PATHS[name],
            headers,
            model,
            f"openai/{name}/{model}",
            attributes,
            slots,
            args.pool,
            args.timeout,
        )
        for model in models
        for name in endpoints
    ]
    print(f"serving {base} as " + ", ".join(handler.spec.interface for handler in handlers), flush=True)
    serve(handlers, endpoint=args.endpoint, pool=args.pool or None)


def list_models(base: str, headers: dict[str, str]) -> list[str]:
    """Return the ids the server lists, waiting while it is still loading."""
    warned = False
    while True:
        try:
            with urllib.request.urlopen(urllib.request.Request(f"{base}/models", headers=headers), timeout=10) as response:
                models = [entry["id"] for entry in json.load(response).get("data", []) if "id" in entry]
            if not models:
                raise SystemExit(f"{base}/models lists no model; name one with --model")
            return models
        except urllib.error.HTTPError as error:
            if error.code < 500:
                raise SystemExit(f"{base}/models returned HTTP {error.code}")
            if not warned:
                print(f"waiting for {base}/models (HTTP {error.code})...", file=sys.stderr, flush=True)
                warned = True
            time.sleep(2)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if not warned:
                print(f"waiting for {base}/models ({error})...", file=sys.stderr, flush=True)
                warned = True
            time.sleep(2)


def forwarder(
    url: str,
    headers: dict[str, str],
    model: str,
    interface: str,
    attributes: dict[str, str],
    slots: int,
    pool: str,
    timeout: float,
):
    @worker(
        interface=interface,
        attributes=attributes,
        max_active_leases=slots,
        pool=pool,
        pass_context=True,
    )
    def forward(payload: bytes, context):
        try:
            body = json.loads(payload)
        except ValueError:
            raise WorkerError("the payload must be valid JSON: an OpenAI request body")
        if not isinstance(body, dict):
            raise WorkerError("the payload must be a JSON object: an OpenAI request body")
        body.setdefault("model", model)
        request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            detail = f"HTTP {error.code}: {error.read().decode(errors='replace')[:300]}"
            # Overload and transient server faults may pass on another worker;
            # rejected 4xx and 501 Not Implemented will not.
            retryable = error.code in (429, 500, 502, 503, 504)
            raise WorkerError(detail, retryable=retryable)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            raise WorkerError(f"server unreachable: {error}", retryable=True)
        if body.get("stream"):
            return events(response, context)
        with response:
            return response.read()

    return forward


def events(response, context):
    """Yield each ``data:`` line of the event stream as one output frame."""
    with response:
        for line in response:
            if context.cancelled:
                return
            if line.startswith(b"data:"):
                yield line


if __name__ == "__main__":
    main()
