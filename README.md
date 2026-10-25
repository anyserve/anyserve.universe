# anyserve.universe

Workers that put real backends behind [anyserve](https://github.com/anyserve/anyserve).
Each directory is one example: a worker that owns the backend's protocol, and a
client that calls it through the anyserve control plane. The control plane
routes opaque bytes; the worker decides what they mean.

| Example | Backend | Payload | Result |
|---|---|---|---|
| [`triton/`](triton) | Triton Inference Server | KServe v2 inference request | Triton's response body |
| [`vllm/`](vllm) | vLLM or any OpenAI-compatible server | chat completions request | response body, or one output frame per server-sent event when streaming |
| [`comfyui/`](comfyui) | ComfyUI | workflow in API format | the first image the workflow saved |
| [`video/`](video) | Ultralytics YOLO | live JPEG frames over a duplex Invoke | annotated frames |

[`setup/`](setup) is a static page that writes a `server.toml` and the matching
`anyserve serve` command from a form, with the client and worker snippets that
connect to it; it is published with GitHub Pages from this repository.

Every example needs a running control plane and the `anyserve` Python package
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
