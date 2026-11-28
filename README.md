# anyserve.universe

Real-world connectors and host-application integrations built on top of
[AnyServe](https://github.com/anyserve/anyserve).
The control plane routes opaque bytes over outbound-only gRPC or WebSocket
(`wss://`); connectors and nodes decide what those bytes mean. Nothing here is
baked into the `anyserve` binary — everything uses the standard Python SDK
(`anyserve-runtime`).

## Inference workers

| Connector | Backend | Interface | Payload | Result |
|---|---|---|---|---|
| [`openai/`](openai) | any OpenAI-compatible server: vLLM, Ollama, LM Studio, llama.cpp, SGLang, embedding servers | `openai/chat/<model>`, `completions`, `embeddings` | OpenAI request JSON | response JSON, or one frame per SSE `data:` line; includes a local OpenAI HTTP `proxy.py` |
| [`triton-inference-server/`](triton-inference-server) | NVIDIA Triton Inference Server | `triton/<model>` | KServe v2 inference JSON | Triton response JSON (auto-discovers `READY` models; includes `batch_scorer` dynamic-batching demo) |
| [`yolo/`](yolo) | Ultralytics YOLO in the worker process | `yolo/<model>` | one JPEG or PNG image | detections JSON + image dimensions (supports per-request `conf`/`labels` metadata and headless `--out`) |
| [`vla/`](vla) | LeRobot SmolVLA vision-language-action policy | `vla/<model>` | robot observation (instruction, joint state, camera frames) | next 50 joint actions (`--probe` smoke test + real-time LeRobot episode replay) |

## Application integrations

| Integration | Host runtime | Calls | What it does |
|---|---|---|---|
| [`comfyui/`](comfyui) | [ComfyUI](https://github.com/comfyanonymous/ComfyUI) custom nodes | `openai/chat/<model>`, `yolo/<model>`, or any interface | Adds `AnyServeConfig`, `AnyServeOpenAIChat`, `AnyServeYOLODetect` (outputs annotated `IMAGE` + inpainting `MASK`), and `AnyServeInvoke` directly to ComfyUI workflows |
| [`ros2/`](ros2) | ROS 2 (`ament_python`, Jazzy/Humble) | `yolo/<model>`, `vla/<model>` | `yolo_node` turns `/camera/image_raw` into `vision_msgs/Detection2DArray` + annotated `/detections/image` with hard frame budgets; `vla_node` runs a 30 Hz `/joint_commands` control loop backed by remote SmolVLA |

## Common shape

- **Workers** register `@worker(interface="<family>/<model>")` and call
  `serve(...)`. Each model gets its own queue, metrics, and routing policy.
  Pass `--pool` for tiering/spillover and `--attr KEY=VALUE` (e.g. `site=edge`,
  `gpu=t4`) for placement with `required_attributes` / `preferred_attributes`.
- **Retry semantics**: an unreachable or overloaded backend (HTTP 429 / 5xx)
  raises `WorkerError(..., retryable=True)` so the control plane reschedules on
  another worker. Invalid inputs / rejected 4xx requests fail without retrying.
- **Zero inbound ports**: workers and clients dial out over `http(s)://` (gRPC)
  or `ws(s)://` (WebSocket).

Requires a running control plane and the `anyserve-runtime` Python package
(Python 3.10+):

```bash
anyserve serve --console-listen 127.0.0.1:3001 --auto-token
pip install anyserve-runtime
```

Workers read a worker-role token from `ANYSERVE_TOKEN` and clients read a
client-role token when control-plane authentication is enabled.

## Measured

One Tesla T4, with client, control plane, worker, and backend on the same host.
"Direct" calls the backend without AnyServe.

| Connector | Backend and load | Direct | Through anyserve |
|---|---|---|---|
| `openai/` | vLLM, Qwen2.5-0.5B-Instruct, 32 output tokens, 32 callers | 95.4 req/s, p50 338 ms | 93.9 req/s, p50 335 ms |
| `openai/` | the same, streaming, time to first token | 19.5 ms | 19.7 ms |
| `triton-inference-server/` | MiniLM cross-encoder on ONNX Runtime, 8 callers | 1,546 req/s, p50 5.2 ms | 1,335 req/s, p50 5.9 ms |
| `yolo/` | YOLO11n on 640×480 video, one camera | 71 fps, 14.0 ms a frame | 62 fps, 15.6 ms a frame |
| `vla/` | SmolVLA on a recorded SO-100 episode at 30 Hz | 585 ms an inference | 588 ms a call |

A call adds roughly 1 to 3 ms on top of the backend.

[`setup/`](setup) is a static form that generates `server.toml` and the matching
`anyserve serve` command. These pages are published at
https://anyserve.github.io/anyserve.universe/ from the READMEs in this
repository, and documented under **Examples** in the
[AnyServe documentation](https://anyserve.github.io/anyserve/).

Licensed under Apache-2.0.
