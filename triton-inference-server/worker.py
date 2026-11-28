"""Put Triton Inference Server models behind anyserve.

The payload is a KServe v2 inference request, the JSON body Triton takes on
``/v2/models/<model>/infer``, and the result is Triton's response body. The
connector does not inspect tensors, so any Triton backend (ONNX Runtime,
TensorRT, PyTorch, Python, vLLM) works unchanged. Triton keeps model loading,
versions, and dynamic batching.

If ``--model`` is omitted, the worker queries Triton's ``/v2/repository/index``
and registers every READY model automatically as ``triton/<model>``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

from anyserve import WorkerError, serve, worker


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


def wait_for_triton(base: str) -> None:
    """Block until Triton reports /v2/health/ready."""
    warned = False
    while True:
        try:
            with urllib.request.urlopen(f"{base}/v2/health/ready", timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if not warned:
                print(f"waiting for Triton at {base} ({error})...", file=sys.stderr, flush=True)
                warned = True
        time.sleep(1)


def discover_ready_models(base: str) -> list[str]:
    """Query Triton's /v2/repository/index and return all READY model names."""
    req = urllib.request.Request(
        f"{base}/v2/repository/index",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        entries = json.load(resp)
    models = sorted({entry["name"] for entry in entries if entry.get("state") == "READY" and "name" in entry})
    if not models:
        raise SystemExit(f"{base}/v2/repository/index lists no READY model; pass --model explicitly")
    return models


def wait_for_model(base: str, model: str) -> None:
    warned = False
    while True:
        try:
            with urllib.request.urlopen(f"{base}/v2/models/{model}/ready", timeout=5) as resp:
                if resp.status == 200:
                    return
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise SystemExit(f"Triton model {model!r} not found at {base}")
            if not warned:
                print(f"waiting for model {model!r} (HTTP {error.code})...", file=sys.stderr, flush=True)
                warned = True
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if not warned:
                print(f"waiting for model {model!r} ({error})...", file=sys.stderr, flush=True)
                warned = True
        time.sleep(1)


def make_infer_handler(
    base: str,
    model: str,
    attributes: dict[str, str],
    slots: int,
    pool: str,
    timeout: float,
):
    infer_url = f"{base}/v2/models/{model}/infer"

    @worker(
        interface=f"triton/{model}",
        attributes=attributes,
        max_active_leases=slots,
        pool=pool,
    )
    def infer(payload: bytes) -> bytes:
        if not payload:
            raise WorkerError("empty KServe v2 inference payload")
        request = urllib.request.Request(
            infer_url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            detail = f"HTTP {error.code}: {error.read().decode(errors='replace')[:300]}"
            # A transient server fault may pass on another Triton; a rejected request will not.
            raise WorkerError(detail, retryable=error.code in (429, 500, 502, 503, 504))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            raise WorkerError(f"triton unreachable: {error}", retryable=True)

    return infer


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Triton Inference Server models through anyserve.")
    parser.add_argument("--endpoint", help="anyserve control plane; default: ANYSERVE_ENDPOINT or http://127.0.0.1:50052")
    parser.add_argument("--triton", default="http://127.0.0.1:8000", help="Triton's HTTP endpoint")
    parser.add_argument("--model", action="append", help="Triton model name (repeatable); default: every READY model")
    parser.add_argument("--concurrency", type=int, default=16, help="requests in flight; give Triton's batcher enough to batch")
    parser.add_argument("--timeout", type=float, default=60.0, help="per-request HTTP timeout to Triton in seconds")
    parser.add_argument("--pool", default="", help="worker pool name for tiering and spillover")
    parser.add_argument("--attr", action="append", default=[], metavar="KEY=VALUE", help="worker attribute for placement, such as site=edge (repeatable)")
    args = parser.parse_args()

    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")

    base = args.triton.rstrip("/")
    wait_for_triton(base)
    models = args.model or discover_ready_models(base)
    for model in models:
        wait_for_model(base, model)

    attributes = parse_attrs(args.attr)
    slots = max(1, args.concurrency // len(models))
    handlers = [
        make_infer_handler(base, model, attributes, slots, args.pool, args.timeout)
        for model in models
    ]
    print(f"serving {base} as " + ", ".join(h.spec.interface for h in handlers), flush=True)
    serve(handlers, endpoint=args.endpoint, pool=args.pool or None)


if __name__ == "__main__":
    main()
