# ComfyUI AnyServe Nodes

Custom nodes that let a [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
workflow call remote workers through [AnyServe](https://github.com/anyserve/anyserve)
— over gRPC or `wss://` with zero open inbound ports. Run ComfyUI on a laptop
or a modest workstation while offloading LLMs/VLMs, YOLO detection/masking,
Triton models, or custom GPU workers to machines behind NAT.

## Included nodes

All nodes live under the **`AnyServe`** category in ComfyUI:

| Node | Calls | Inputs → Outputs | Use case |
|---|---|---|---|
| **AnyServe Config** (`AnyServeConfig`) | — | endpoint, optional token/namespace, timeout, required/preferred attributes, priority → `ANYSERVE_CONFIG` | Share connection and placement rules across nodes (leave `token` blank to read `ANYSERVE_TOKEN` from the environment) |
| **AnyServe OpenAI Chat / VLM** (`AnyServeOpenAIChat`) | `openai/chat/<model>` | model, prompt, system prompt, optional `IMAGE` → `STRING` | Expand short prompts via remote vLLM/Ollama, or caption an input image with a remote VLM and feed into `CLIPTextEncode` |
| **AnyServe YOLO Detect & Mask** (`AnyServeYOLODetect`) | `yolo/<model>` | `IMAGE` batch, model, confidence, label filter, mask padding → `annotated_image`, `MASK`, `detections_json`, `count` | Detect people/objects on a remote GPU and wire `MASK` straight into `VAEEncodeForInpaint` or `ImageCompositeMasked` |
| **AnyServe Invoke (Generic)** (`AnyServeInvoke`) | any `<interface>` | interface, text or PNG/JPEG `IMAGE` → `text`, `IMAGE` | Bridge any custom Python or Rust AnyServe worker (upscaler, segmenter, Triton, etc.) into a ComfyUI graph |

## Install into ComfyUI

Copy (or symlink) this directory into `ComfyUI/custom_nodes/comfyui-anyserve`
and install its requirements into ComfyUI's Python environment:

```bash
cp -r comfyui /path/to/ComfyUI/custom_nodes/comfyui-anyserve
pip install -r /path/to/ComfyUI/custom_nodes/comfyui-anyserve/requirements.txt
```

Set `ANYSERVE_ENDPOINT` (default `http://127.0.0.1:50052`) and, when auth is
enabled, `ANYSERVE_TOKEN` (a client token) in ComfyUI's environment so tokens
stay out of saved workflow JSON files.

## Run in Docker (CPU ComfyUI + remote GPU workers)

`Dockerfile` builds a lightweight CPU-only ComfyUI container with
`comfyui-anyserve` preinstalled — ideal when inference runs on remote GPU
workers over AnyServe:

```bash
docker build -t comfyui-anyserve ./comfyui
docker run --rm -p 8188:8188 \
  -e ANYSERVE_ENDPOINT=http://host.docker.internal:50052 \
  -e ANYSERVE_TOKEN="<client-token>" \
  comfyui-anyserve
```

## Example workflow

[`workflow_api.json`](workflow_api.json) chains `LoadImage` →
`AnyServeYOLODetect` → `SaveImage` (annotated image) and `MaskToImage` →
`SaveImage` (binary detection mask), alongside `AnyServeOpenAIChat` →
`AnyServeInvoke`:

```bash
# Start AnyServe + yolo/worker.py + openai/worker.py first, then submit the workflow:
curl -s http://127.0.0.1:8188/prompt \
  -H 'Content-Type: application/json' \
  -d "{\"prompt\": $(cat comfyui/workflow_api.json)}"
```

## Container verification

Verified inside a CPU-only ComfyUI Docker container calling authenticated
AnyServe v0.7.0 GPU workers on a Tesla T4 over a Docker bridge network:

- All 4 nodes (`AnyServeConfig`, `AnyServeOpenAIChat`, `AnyServeYOLODetect`,
  `AnyServeInvoke`) register cleanly in `/object_info`.
- Submitting `workflow_api.json` to `/prompt` executes `AnyServeYOLODetect`
  against `yolo/yolo11n`, saving both the annotated PNG and a binary detection
  mask PNG, while chaining `AnyServeOpenAIChat` (Ollama `qwen2.5:0.5b`) into
  `AnyServeInvoke`.
