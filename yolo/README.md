# YOLO

`worker.py` runs an Ultralytics YOLO model in the worker process on the
interface `yolo/<model>`, such as `yolo/yolo11n`. One call is one image: the
payload is the encoded image, JPEG or PNG, and the result is the detections.

```json
{"detections": [{"label": "person", "confidence": 0.91, "box": [48.2, 398.1, 245.0, 902.7]}], "inference_ms": 10.5}
```

`client.py` plays a video or a camera feed against it, one call per frame. A
frame that misses `--budget-ms` is dropped instead of delaying the ones behind it.

```bash
pip install ultralytics
python worker.py --model yolo11n.pt
python client.py --source demo.mp4            # or --source 0 for a camera; --show draws the boxes
```

The worker takes one image at a time. Cameras share a worker frame by frame,
and a second worker adds capacity without a change to the clients.

On a Tesla T4 with YOLO11n and 640×480 video:

| Setup | Each camera |
|---|---|
| the model called in process, no anyserve | 71 fps, 14.0 ms a frame |
| one camera, one worker | 62 fps, p50 15.6 ms |
| two cameras, one worker | 37 fps, p50 26.9 ms |
| two cameras, two workers on the same GPU | 56 fps, p50 17.0 ms |
