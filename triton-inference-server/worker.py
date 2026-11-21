"""Put a Triton Inference Server model behind anyserve.

The payload is a KServe v2 inference request, the JSON body Triton takes on
``/v2/models/<model>/infer``, and the result is Triton's response body. The
connector does not read the tensors, so any model works unchanged. Triton keeps
model loading, versions and dynamic batching.
"""
from __future__ import annotations
import argparse
import urllib.error
import urllib.request

from anyserve import WorkerError, serve, worker


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a Triton model through anyserve.")
    parser.add_argument("--endpoint", help="anyserve control plane; default: ANYSERVE_ENDPOINT or http://127.0.0.1:50052")
    parser.add_argument("--triton", default="http://127.0.0.1:8000", help="Triton's HTTP endpoint")
    parser.add_argument("--model", required=True, help="Triton model name; the interface is triton/<model>")
    parser.add_argument("--concurrency", type=int, default=16, help="requests in flight; give Triton's batcher enough to batch")
    parser.add_argument("--attr", action="append", default=[], metavar="KEY=VALUE", help="worker attribute for placement, such as site=edge (repeatable)")
    args = parser.parse_args()
    url = f"{args.triton.rstrip('/')}/v2/models/{args.model}"
    urllib.request.urlopen(f"{url}/ready", timeout=10).close()  # raises unless Triton has the model loaded

    @worker(interface=f"triton/{args.model}", attributes=dict(item.partition("=")[::2] for item in args.attr),
            max_active_leases=args.concurrency)
    def infer(payload: bytes) -> bytes:
        request = urllib.request.Request(f"{url}/infer", data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            detail = f"HTTP {error.code}: {error.read().decode(errors='replace')[:300]}"
            # A server fault may pass on another Triton; a rejected request will not.
            raise WorkerError(detail, retryable=error.code >= 500)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            raise WorkerError(f"triton unreachable: {error}", retryable=True)

    serve(infer, args.endpoint)


if __name__ == "__main__":
    main()
