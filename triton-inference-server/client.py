"""Invoke a Triton Inference Server model through anyserve (single call or concurrent benchmark)."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from pathlib import Path

from anyserve import AnyserveClient, ExecutionOptions

DEFAULT_PAYLOAD = {
    "inputs": [
        {
            "name": "FEATURES",
            "shape": [1, 4],
            "datatype": "FP32",
            "data": [0.2, 0.5, 0.8, 1.0],
        }
    ]
}


def parse_attrs(items: list[str]) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"invalid attribute {item!r}: expected KEY=VALUE")
        key, value = item.split("=", 1)
        attrs[key] = value
    return attrs


def main() -> None:
    parser = argparse.ArgumentParser(description="Send KServe v2 inference requests to Triton through anyserve.")
    parser.add_argument("--endpoint", help="default: ANYSERVE_ENDPOINT or http://127.0.0.1:50052")
    parser.add_argument("--model", default="batch_scorer", help="Triton model name; interface is triton/<model>")
    parser.add_argument("--input", help="inline KServe v2 JSON string")
    parser.add_argument("--input-file", help="path to a KServe v2 JSON file")
    parser.add_argument("--requests", type=int, default=1, help="number of requests to run (default: 1)")
    parser.add_argument("--concurrency", type=int, default=1, help="parallel client threads for batching benchmark")
    parser.add_argument("--timeout-ms", type=int, default=60_000)
    parser.add_argument("--attr", action="append", default=[], metavar="KEY=VALUE", help="preferred worker attribute (repeatable)")
    parser.add_argument("--require", action="append", default=[], metavar="KEY=VALUE", help="required worker attribute (repeatable)")
    args = parser.parse_args()

    if args.input_file:
        payload = Path(args.input_file).read_bytes()
    elif args.input:
        payload = args.input.encode()
    else:
        payload = json.dumps(DEFAULT_PAYLOAD).encode()

    interface = f"triton/{args.model}"
    options = ExecutionOptions(
        total_timeout_ms=args.timeout_ms,
        preferred_attributes=parse_attrs(args.attr),
        required_attributes=parse_attrs(args.require),
    )

    if args.requests == 1 and args.concurrency == 1:
        client = AnyserveClient(args.endpoint)
        started = time.perf_counter()
        raw = client.call(interface, payload, options=options)
        elapsed_ms = (time.perf_counter() - started) * 1000
        reply = json.loads(raw)
        print(json.dumps(reply, indent=2))
        print(f"# {interface}: {elapsed_ms:.2f} ms")
        return

    latencies: list[float] = []
    started = time.perf_counter()

    def run_worker(count: int) -> tuple[list[float], bytes]:
        c = AnyserveClient(args.endpoint)
        times: list[float] = []
        last = b""
        for _ in range(count):
            t0 = time.perf_counter()
            last = c.call(interface, payload, options=options)
            times.append((time.perf_counter() - t0) * 1000)
        return times, last

    workers = max(1, min(args.concurrency, args.requests))
    base_n, rem = divmod(args.requests, workers)
    counts = [base_n + (1 if i < rem else 0) for i in range(workers)]
    sample = b""
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for times, last in pool.map(run_worker, counts):
            latencies.extend(times)
            sample = last

    wall = time.perf_counter() - started
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
    print(
        f"{len(latencies)} requests across {workers} callers in {wall:.2f}s: "
        f"{len(latencies) / wall:.1f} req/s, p50 {p50:.2f} ms, p95 {p95:.2f} ms, max {latencies[-1]:.2f} ms"
    )
    print("sample output:", json.dumps(json.loads(sample).get("outputs", [])))


if __name__ == "__main__":
    main()
