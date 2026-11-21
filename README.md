# anyserve.universe

Connectors that put real inference backends behind [anyserve](https://github.com/anyserve/anyserve).
The control plane routes opaque bytes; a connector decides what they mean.
Nothing here is part of the anyserve binary: a connector is an ordinary worker
on the Python SDK.

| Connector | Backend | Interface | Payload | Result |
|---|---|---|---|---|
| [`openai/`](openai) | any OpenAI-compatible server: vLLM, Ollama, LM Studio, llama.cpp, embedding servers | `openai/chat/<model>`, also `completions` and `embeddings` | the OpenAI request body | the response body, or one frame per server-sent event |
| [`triton-inference-server/`](triton-inference-server) | Triton Inference Server | `triton/<model>` | a KServe v2 inference request | Triton's response body |
| [`yolo/`](yolo) | Ultralytics YOLO, in the worker process | `yolo/<model>` | one JPEG or PNG image | the detections as JSON |
| [`vla/`](vla) | SmolVLA, a vision-language-action policy, in the worker process | `vla/<model>` | one robot observation: instruction, joint state, camera frames | the next 50 actions |

Every connector has the same shape:

- **`worker.py`** is one file. It registers `@worker(interface="<family>/<model>")`
  on a function that takes the payload and returns the result, then calls
  `serve(...)`. The model is part of the interface name, so each model has its
  own queue and metrics. `--attr KEY=VALUE` describes the deployment, such as
  `site=edge`, for placement with `preferred_attributes`.
- **`client.py`** is one file that calls the interface through the control plane.
- A backend that is unreachable or failing is reported with
  `WorkerError(..., retryable=True)`, so the control plane can run the job on
  another worker. A request the backend rejects fails without a retry.

They need a running control plane and the `anyserve` Python package (Python 3.12+):

```bash
anyserve serve
pip install anyserve-runtime      # or pip install ./clients/python from the anyserve checkout
```

Workers read a worker-role token from `ANYSERVE_TOKEN` and clients a client
token, both only when the control plane has auth enabled.

## Measured

One Tesla T4, with the client, control plane, worker and backend on the same
host. "Direct" calls the backend without anyserve.

| Connector | Backend and load | Direct | Through anyserve |
|---|---|---|---|
| `openai/` | vLLM, Qwen2.5-0.5B-Instruct, 32 output tokens, 32 callers | 95.4 req/s, p50 338 ms | 93.9 req/s, p50 335 ms |
| `openai/` | the same, streaming, time to first token | 19.5 ms | 19.7 ms |
| `triton-inference-server/` | MiniLM cross-encoder on the ONNX Runtime backend, 8 callers | 1,546 req/s, p50 5.2 ms | 1,335 req/s, p50 5.9 ms |
| `yolo/` | YOLO11n on 640×480 video, one camera | 71 fps, 14.0 ms a frame | 62 fps, 15.6 ms a frame |
| `vla/` | SmolVLA on a recorded SO-100 episode at 30 Hz | 585 ms an inference | 588 ms a call |

A call costs 1 to 3 ms on top of the backend. That is invisible behind a
language model or a policy and is about a quarter of the throughput of a
backend that answers in 3 ms.

[`setup/`](setup) is a page that writes a `server.toml` and the matching
`anyserve serve` command from a form. These pages are published at
https://anyserve.github.io/anyserve.universe/ from the READMEs in this
repository.

Licensed under Apache-2.0.
