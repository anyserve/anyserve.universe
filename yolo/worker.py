"""Run an Ultralytics YOLO model behind anyserve.

One call is one image. The payload is an encoded JPEG or PNG image, and the
result is JSON detections plus image dimensions:

    {
      "detections": [{"label": "person", "confidence": 0.91, "box": [x1, y1, x2, y2]}, ...],
      "image": {"width": 640, "height": 480},
      "inference_ms": 7.9
    }

Clients may optionally override confidence or filter labels per request via
``ExecutionOptions(metadata={"conf": "0.4", "labels": "person,car"})``.
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


def parse_attrs(items: list[str]) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"invalid --attr {item!r}: expected KEY=VALUE")
        key, value = item.split("=", 1)
        if not key:
            raise SystemExit(f"invalid --attr {item!r}: key must not be empty")
        attrs[key] = value
    return attrs


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a YOLO model through anyserve.")
    parser.add_argument("--endpoint", help="anyserve control plane; default: ANYSERVE_ENDPOINT or http://127.0.0.1:50052")
    parser.add_argument("--model", default="yolo11n.pt", help="weights to load; the interface is yolo/<name>, such as yolo/yolo11n")
    parser.add_argument("--conf", type=float, default=0.25, help="default lowest confidence to report")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size (default: 640)")
    parser.add_argument("--device", default=None, help="torch device such as cuda:0 or cpu (default: auto)")
    parser.add_argument("--pool", default="", help="worker pool name for tiering and spillover")
    parser.add_argument("--attr", action="append", default=[], metavar="KEY=VALUE", help="worker attribute for placement, such as site=edge (repeatable)")
    args = parser.parse_args()

    model = YOLO(args.model)
    predict_kwargs: dict = {"imgsz": args.imgsz, "verbose": False}
    if args.device is not None:
        predict_kwargs["device"] = args.device
    # Warm up weights and CUDA kernels before taking calls.
    model.predict(numpy.zeros((args.imgsz, args.imgsz, 3), numpy.uint8), **predict_kwargs)

    interface = f"yolo/{Path(args.model).stem}"
    attributes = parse_attrs(args.attr)

    # One image at a time: the model runs on one device.
    @worker(
        interface=interface,
        attributes=attributes,
        max_active_leases=1,
        pool=args.pool,
        pass_context=True,
    )
    def detect(image: bytes, context) -> bytes:
        frame = cv2.imdecode(numpy.frombuffer(image, numpy.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise WorkerError("the payload is not a JPEG or PNG image")
        meta = getattr(context, "metadata", None) or {}
        conf = args.conf
        if "conf" in meta:
            try:
                conf = float(meta["conf"])
            except ValueError:
                raise WorkerError(f"invalid metadata conf={meta['conf']!r}")
        allowed = None
        if meta.get("labels"):
            allowed = {label.strip().lower() for label in meta["labels"].split(",") if label.strip()}

        started = time.perf_counter()
        result = model.predict(frame, conf=conf, **predict_kwargs)[0]
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        boxes = result.boxes
        detections = []
        if boxes is not None:
            for box, confidence, label_idx in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
                label = str(model.names[int(label_idx)])
                if allowed is not None and label.lower() not in allowed:
                    continue
                detections.append(
                    {
                        "label": label,
                        "confidence": round(confidence, 3),
                        "box": [round(value, 1) for value in box],
                    }
                )
        height, width = frame.shape[:2]
        return json.dumps(
            {
                "detections": detections,
                "image": {"width": width, "height": height},
                "inference_ms": elapsed_ms,
            }
        ).encode()

    print(f"serving {args.model} as {interface}", flush=True)
    serve(detect, endpoint=args.endpoint, pool=args.pool or None)


if __name__ == "__main__":
    main()
