# Triton Inference Server

`worker.py` serves one Triton model on the interface `triton/<model>`. The
payload is the KServe v2 inference request Triton takes on
`/v2/models/<model>/infer` and the result is Triton's response body, so the
connector needs no knowledge of the model's tensors. Triton keeps model
loading, versions and dynamic batching.

```bash
python worker.py --triton http://127.0.0.1:8000 --model MODEL
```

```python
import json
from anyserve import AnyserveClient

request = {"inputs": [{"name": "input", "shape": [1, 4], "datatype": "FP32", "data": [0.1, 0.2, 0.3, 0.4]}]}
reply = json.loads(AnyserveClient().call("triton/MODEL", json.dumps(request).encode()))
print(reply["outputs"])
```

`--concurrency` (default 16) is how many requests the worker keeps in flight;
Triton's batcher can only batch what arrives together. A Triton that is
unreachable or answers HTTP 5xx is retryable; a request it rejects is not. Run
one worker next to each Triton server.

On a Tesla T4, a MiniLM cross-encoder (22M parameters) on the ONNX Runtime
backend with dynamic batching, one query-passage pair a request:

| Callers | Direct | Through anyserve |
|---|---|---|
| 1 | 344 req/s, p50 2.9 ms | 248 req/s, p50 4.0 ms |
| 8 | 1,546 req/s, p50 5.2 ms | 1,335 req/s, p50 5.9 ms |
| 32 | 2,043 req/s, p50 15.9 ms | 1,538 req/s, p50 20.7 ms |

A model this fast shows the cost of a call: about 1 ms, which is a quarter of
the throughput with one caller.
