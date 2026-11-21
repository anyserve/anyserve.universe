"""Put any OpenAI-compatible inference server behind anyserve.

vLLM, Ollama, LM Studio, llama.cpp's server, SGLang and embedding servers all
speak the same HTTP API, so one connector covers them. The connector asks the
server which models it has and registers one interface per model and endpoint:

    openai/chat/<model>          POST <base>/chat/completions
    openai/completions/<model>   POST <base>/completions
    openai/embeddings/<model>    POST <base>/embeddings

The payload is the request body as the OpenAI API takes it. The result is the
response body, or one output frame per server-sent event line when the request
asks to stream. The server keeps the model, its cache and its batching;
anyserve adds placement across servers, deadlines and failover.
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
    parser.add_argument("--serve", default="chat", help="comma-separated endpoints: chat, completions, embeddings")
    parser.add_argument("--concurrency", type=int, default=16, help="requests in flight on this server, across its interfaces")
    parser.add_argument("--attr", action="append", default=[], metavar="KEY=VALUE", help="worker attribute such as site=edge (repeatable)")
    parser.add_argument("--pool", help="scheduling pool this server belongs to")
    parser.add_argument("--wait", type=float, default=600, help="seconds to wait for the server to answer before giving up")
    parser.add_argument("--timeout", type=float, default=600, help="seconds one request may take")
    parser.add_argument("--worker-id")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    endpoints = [name.strip() for name in args.serve.split(",") if name.strip()]
    unknown = [name for name in endpoints if name not in PATHS]
    if unknown or not endpoints:
        parser.error(f"--serve takes {', '.join(PATHS)}; got {args.serve!r}")
    attributes = dict(parse_attr(parser, item) for item in args.attr)
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    listed = wait_for_models(base, headers, args.wait, named=bool(args.model))
    models = args.model or listed
    if not models:
        parser.error(f"{base}/models lists no model; name one with --model")

    handlers = []
    slots = max(1, args.concurrency // (len(models) * len(endpoints)))
    for model in models:
        for name in endpoints:
            handlers.append(forwarder(base + PATHS[name], headers, args.timeout, model,
                                      interface=f"openai/{name}/{model}", attributes=attributes, slots=slots))
    print(f"serving {len(handlers)} interface(s) from {base}: " + ", ".join(h.spec.interface for h in handlers), flush=True)
    serve(handlers, endpoint=args.endpoint, worker_id=args.worker_id, pool=args.pool)


def parse_attr(parser: argparse.ArgumentParser, item: str) -> tuple[str, str]:
    key, separator, value = item.partition("=")
    if not separator or not key:
        parser.error(f"--attr takes KEY=VALUE; got {item!r}")
    return key, value


def wait_for_models(base: str, headers: dict[str, str], wait: float, *, named: bool) -> list[str]:
    """Return the ids the server lists, once it answers. A server still loading its model is waited for."""
    deadline = time.monotonic() + wait
    while True:
        try:
            with urllib.request.urlopen(urllib.request.Request(f"{base}/models", headers=headers), timeout=10) as response:
                return [entry["id"] for entry in json.load(response).get("data", []) if entry.get("id")]
        except urllib.error.HTTPError as error:
            if error.code == 404 and named:
                return []  # the server is up but has no model list; --model names what to serve
            # The server is up and refused: waiting will not change its answer.
            raise SystemExit(f"{base}/models returned HTTP {error.code}: {error.read().decode(errors='replace')[:200]}")
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as error:
            if time.monotonic() >= deadline:
                raise SystemExit(f"{base}/models did not answer within {wait:g}s: {error}")
            time.sleep(2)


def forwarder(url: str, headers: dict[str, str], timeout: float, model: str, *, interface: str,
              attributes: dict[str, str], slots: int):
    @worker(interface=interface, attributes=attributes, max_active_leases=slots)
    def forward(payload: bytes):
        try:
            body = json.loads(payload)
        except ValueError:
            body = None
        if not isinstance(body, dict):
            raise WorkerError("the payload must be a JSON object: an OpenAI request body")
        body.setdefault("model", model)
        request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            detail = f"HTTP {error.code}: {error.read().decode(errors='replace')[:300]}"
            # Overload and server faults may pass on another server; a rejected request will not.
            raise WorkerError(detail, retryable=error.code == 429 or error.code >= 500)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            raise WorkerError(f"server unreachable: {error}", retryable=True)
        if body.get("stream"):
            return stream_events(response)
        with response:
            return response.read()

    return forward


def stream_events(response):
    """Yield each ``data:`` line of the event stream as one output frame."""
    with response:
        for line in response:
            if line.startswith(b"data:"):
                yield line.rstrip(b"\r\n") + b"\n"


if __name__ == "__main__":
    main()
