# YOLO

`worker.py` runs an Ultralytics YOLO model in the worker process, on live
video. A duplex Invoke on the interface `yolo.detect.v1` carries JPEG frames to
the worker and annotated frames back, one invocation per batch of frames. The client processes each frame's
result before sending the next one and rotates invocations before the input
replay budget fills. Video processing needs OpenCV and Ultralytics.

```bash
pip install -r requirements.txt
python worker.py --device auto            # cuda, mps or cpu
python client.py --source 0               # a webcam, or a path to an mp4
python client.py --source demo.mp4 --headless --max-frames 120
```

Press `q` in the OpenCV window to stop capture and finish the current
invocation. Closing an active invocation cancels it. `common.py` holds the
application framing; anyserve itself transports opaque bytes.
