# Triton Inference Server

`worker.py` serves one Triton model. The payload is the KServe v2 inference
request that Triton accepts on `/v2/models/<name>/infer`, and the result is
Triton's response body, so the worker needs no knowledge of the model's
tensors. Triton keeps model loading, versions and dynamic batching; anyserve
adds placement, deadlines and failover across hosts.

```bash
python worker.py --triton http://127.0.0.1:8000 --model MODEL
```

Call it with a KServe v2 request on the interface `triton/MODEL`:

```python
import json
from anyserve import AnyserveClient

client = AnyserveClient("http://127.0.0.1:50052")
request = {"inputs": [{"name": "input", "shape": [1, 4], "datatype": "FP32", "data": [0.1, 0.2, 0.3, 0.4]}]}
response = json.loads(client.call("triton/MODEL", json.dumps(request).encode()))
print(response["outputs"])
```

Set `--concurrency` to the model's instance count so Triton's batcher sees
enough concurrent requests. Run one worker next to each Triton server and give
them distinguishing `attributes` to steer placement with
`preferred_attributes` and `spillover_after_ms`. When Triton is unreachable the
worker reports a retryable failure and the control plane can run the job on
another worker.
