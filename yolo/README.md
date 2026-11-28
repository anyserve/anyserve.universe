# YOLO

`worker.py` runs an Ultralytics YOLO model in the worker process on the
interface `yolo/<model>`, such as `yolo/yolo11n`. One call is one image: the
payload is an encoded JPEG or PNG image, and the result is JSON detections plus
the image dimensions:

```json
{
  "detections": [{"label": "person", "confidence": 0.91, "box": [48.2, 398.1, 245.0, 902.7]}],
  "image": {"width": 640, "height": 480},
  "inference_ms": 10.5
}
```

Clients can optionally override the confidence threshold or class filter per
request via `ExecutionOptions(metadata={"conf": "0.4", "labels": "person,car"})`.
This same `yolo/<model>` interface is consumed directly by the
[`comfyui/`](../comfyui/) custom nodes and the [`ros2/`](../ros2/) perception node.

## Quickstart

```bash
pip install anyserve-runtime ultralytics

# Start the GPU worker
python worker.py --model yolo11n.pt --device cuda:0 --attr gpu=t4

# Single image (prints JSON and saves an annotated image headless)
python client.py --source photo.jpg --out annotated.jpg --labels person,car --conf 0.3 --json

# Video file or webcam (drops frames that miss --budget-ms instead of falling behind)
python client.py --source demo.mp4 --out annotated.mp4 --budget-ms 500
python client.py --source 0 --show
```

## Worker options

| Option | Meaning |
|---|---|
| `--endpoint` | control plane URL; default `ANYSERVE_ENDPOINT` or `http://127.0.0.1:50052` |
| `--model` | Ultralytics weights file; interface is `yolo/<stem>` (default `yolo11n.pt`) |
| `--conf` | default confidence threshold (default `0.25`) |
| `--imgsz` | inference image size (default `640`) |
| `--device` | torch device such as `cuda:0` or `cpu` (default: auto) |
| `--pool` | worker pool name for tiering and spillover |
| `--attr KEY=VALUE` | worker attribute for placement (repeatable) |

The worker takes one image at a time. Multiple cameras share a worker frame by
frame, and adding a second worker scales capacity with no client changes.

## Measured

On a Tesla T4 with YOLO11n and 640×480 video:

| Setup | Each camera |
|---|---|
| the model called in process, no anyserve | 71 fps, 14.0 ms a frame |
| one camera, one worker | 62 fps, p50 15.6 ms |
| two cameras, one worker | 37 fps, p50 26.9 ms |
| two cameras, two workers on the same GPU | 56 fps, p50 17.0 ms |
