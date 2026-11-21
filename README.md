# anyserve.universe

Connectors that put real backends behind [anyserve](https://github.com/anyserve/anyserve).
Each directory is one connector: a worker that owns the backend's protocol, and a
client that calls it through the anyserve control plane. The control plane
routes opaque bytes; the connector decides what they mean. Nothing here is part
of the anyserve binary: a connector is an ordinary worker on the SDK.

| Connector | Backend | Payload | Result |
|---|---|---|---|
| [`openai/`](openai) | any OpenAI-compatible server: vLLM, Ollama, LM Studio, llama.cpp, embedding servers | chat completions, completions or embeddings request | response body, or one output frame per server-sent event when streaming |
| [`triton-inference-server/`](triton-inference-server) | Triton Inference Server | KServe v2 inference request | Triton's response body |
| [`yolo/`](yolo) | Ultralytics YOLO | live JPEG frames over a duplex Invoke | annotated frames |
| [`vla/`](vla) | a vision-language-action policy: SmolVLA, or a simulated one that needs no GPU | one robot observation: instruction, state, camera frames | a chunk of actions for the next control ticks |

[`setup/`](setup) is a page that writes a `server.toml` and the matching
`anyserve serve` command from a form, with the client and worker snippets that
connect to it. These pages are published at https://anyserve.github.io/anyserve.universe/
from the READMEs in this repository.

Every connector needs a running control plane and the `anyserve` Python package
(Python 3.12+):

```bash
anyserve serve
pip install anyserve-runtime      # or pip install ./clients/python from the anyserve checkout
```

The workers share one shape: `@worker(interface=...)` on a function that takes
the payload bytes and returns bytes or yields frames, then `serve(handler,
endpoint)`. A backend that is unreachable is reported with
`WorkerError(..., retryable=True)` so the control plane can run the job on
another worker; a request the backend rejects fails without a retry. Workers
read a worker-role token from `ANYSERVE_TOKEN`, clients read a client token.

Licensed under Apache-2.0.
