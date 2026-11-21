"""Detect objects in a video or a camera feed, one anyserve call per frame."""
from __future__ import annotations
import argparse
import json
import time

import cv2

from anyserve import AnyserveClient, ExecutionOptions


def main() -> None:
    parser = argparse.ArgumentParser(description="Send video frames to a YOLO model through anyserve.")
    parser.add_argument("--endpoint", help="default: ANYSERVE_ENDPOINT or http://127.0.0.1:50052")
    parser.add_argument("--model", default="yolo11n", help="the interface is yolo/<model>")
    parser.add_argument("--source", default="0", help="a camera index or a video file")
    parser.add_argument("--frames", type=int, default=0, help="stop after this many frames; 0 plays the whole source")
    parser.add_argument("--budget-ms", type=int, default=1000, help="time one frame may take before it is dropped")
    parser.add_argument("--show", action="store_true", help="draw the boxes in a window; q closes it")
    args = parser.parse_args()
    capture = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not capture.isOpened():
        raise SystemExit(f"could not open {args.source!r}")
    client = AnyserveClient(args.endpoint)
    options = ExecutionOptions(queue_timeout_ms=args.budget_ms, total_timeout_ms=args.budget_ms, max_attempts=1)
    latencies, dropped = [], 0
    started = time.perf_counter()
    while not args.frames or len(latencies) + dropped < args.frames:
        ok, frame = capture.read()
        if not ok:
            break
        sent = time.perf_counter()
        try:
            reply = json.loads(client.call(f"yolo/{args.model}", cv2.imencode(".jpg", frame)[1].tobytes(), options=options))
        except RuntimeError:
            dropped += 1  # a late frame is worth nothing to live video
            continue
        latencies.append((time.perf_counter() - sent) * 1000)
        if args.show:
            for detection in reply["detections"]:
                x1, y1, x2, y2 = map(int, detection["box"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(frame, detection["label"], (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
            cv2.imshow("yolo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    if not latencies:
        raise SystemExit(f"no frame was answered; {dropped} dropped")
    latencies.sort()
    print(f"{len(latencies)} frames at {len(latencies) / (time.perf_counter() - started):.1f} fps, {dropped} dropped, "
          f"latency p50 {latencies[len(latencies) // 2]:.1f} ms, max {latencies[-1]:.1f} ms; "
          f"last frame: {len(reply['detections'])} objects, inference {reply['inference_ms']} ms")


if __name__ == "__main__":
    main()
