# Triton Inference Server

`worker.py` puts [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server)
models behind anyserve on `triton/<model>`. The payload is the KServe v2
inference request Triton takes on `/v2/models/<model>/infer` and the result is
Triton's response body, so the connector works unchanged across ONNX Runtime,
TensorRT, PyTorch, Python, and vLLM backends. Triton keeps model loading,
versions, and dynamic batching.

If `--model` is omitted, `worker.py` waits for Triton's `/v2/health/ready`,
queries `/v2/repository/index`, and registers every `READY` model automatically.

## Quickstart (with included demo model)

`demo_model_repository/batch_scorer` is a minimal Triton Python-backend model
configured with `dynamic_batching` (`preferred_batch_size: [4, 8, 16]`) that
returns both a score and the batch size Triton formed:

```bash
# 1. Start Triton with the included demo repository
docker run --rm -p 8000:8000 \
  -v "$PWD/demo_model_repository:/models:ro" \
  nvcr.io/nvidia/tritonserver:26.01-py3 \
  tritonserver --model-repository=/models

# 2. Connect the outbound worker (auto-discovers all READY models)
python worker.py --triton http://127.0.0.1:8000 --concurrency 16 --attr site=lab

# 3. Single inference call or concurrent dynamic-batching benchmark
python client.py --model batch_scorer
python client.py --model batch_scorer --requests 200 --concurrency 16
```

## Calling from Python

```python
import json
from anyserve import AnyserveClient

request = {
    "inputs": [
        {"name": "FEATURES", "shape": [1, 4], "datatype": "FP32", "data": [0.2, 0.5, 0.8, 1.0]}
    ]
}
reply = json.loads(AnyserveClient().call("triton/batch_scorer", json.dumps(request).encode()))
print(reply["outputs"])
```

## Worker options

| Option | Meaning |
|---|---|
| `--endpoint` | control plane URL; default `ANYSERVE_ENDPOINT` or `http://127.0.0.1:50052` |
| `--triton` | Triton HTTP base URL; default `http://127.0.0.1:8000` |
| `--model` | Triton model name (repeatable); default: every `READY` model in `/v2/repository/index` |
| `--concurrency` | requests in flight; give Triton's dynamic batcher enough concurrent requests to batch (default 16) |
| `--timeout` | per-request HTTP timeout to Triton in seconds (default 60) |
| `--pool` | worker pool name for tiering and spillover |
| `--attr KEY=VALUE` | worker attribute for placement (repeatable) |

An unreachable Triton or HTTP 429/5xx response is reported as retryable; a
rejected 4xx request fails without a retry. Run one worker next to each Triton
instance.

## Measured

On a Tesla T4, a MiniLM cross-encoder (22M parameters) on the ONNX Runtime
backend with dynamic batching, one query-passage pair a request:

| Callers | Direct | Through anyserve |
|---|---|---|
| 1 | 344 req/s, p50 2.9 ms | 248 req/s, p50 4.0 ms |
| 8 | 1,546 req/s, p50 5.2 ms | 1,335 req/s, p50 5.9 ms |
| 32 | 2,043 req/s, p50 15.9 ms | 1,538 req/s, p50 20.7 ms |

A model this fast shows the cost of a call: about 1 ms, which is a quarter of
the throughput with one caller.
