"""Run an Ultralytics YOLO model behind anyserve.

One call is one image. The payload is the encoded image, JPEG or PNG, and the
result is the detections:

    {"detections": [{"label": "person", "confidence": 0.91, "box": [x1, y1, x2, y2]}, ...], "inference_ms": 7.9}
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy
from ultralytics import YOLO

from anyserve import WorkerError, serve, worker


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a YOLO model through anyserve.")
    parser.add_argument("--endpoint", help="anyserve control plane; default: ANYSERVE_ENDPOINT or http://127.0.0.1:50052")
    parser.add_argument("--model", default="yolo11n.pt", help="weights to load; the interface is yolo/<name>, such as yolo/yolo11n")
    parser.add_argument("--conf", type=float, default=0.25, help="lowest confidence to report")
    parser.add_argument("--attr", action="append", default=[], metavar="KEY=VALUE", help="worker attribute for placement, such as site=edge (repeatable)")
    args = parser.parse_args()
    model = YOLO(args.model)
    model.predict(numpy.zeros((640, 640, 3), numpy.uint8), verbose=False)  # load the weights before taking calls

    # One image at a time: the model runs on one device.
    @worker(interface=f"yolo/{Path(args.model).stem}", attributes=dict(item.partition("=")[::2] for item in args.attr),
            max_active_leases=1)
    def detect(image: bytes) -> bytes:
        frame = cv2.imdecode(numpy.frombuffer(image, numpy.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise WorkerError("the payload is not a JPEG or PNG image")
        started = time.perf_counter()
        boxes = model.predict(frame, conf=args.conf, verbose=False)[0].boxes
        detections = [{"label": model.names[int(label)], "confidence": round(confidence, 3), "box": [round(value, 1) for value in box]}
                      for box, confidence, label in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist())]
        return json.dumps({"detections": detections, "inference_ms": round((time.perf_counter() - started) * 1000, 1)}).encode()

    serve(detect, args.endpoint)


if __name__ == "__main__":
    main()
