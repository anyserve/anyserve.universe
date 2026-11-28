"""Detect objects in an image, video, or camera feed through anyserve."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from anyserve import AnyserveClient, ExecutionOptions

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_attrs(items: list[str]) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"invalid attribute {item!r}: expected KEY=VALUE")
        key, value = item.split("=", 1)
        attrs[key] = value
    return attrs


def draw_detections(frame, detections: list[dict]) -> None:
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["box"])
        label = f"{detection['label']} {detection['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 0),
            2,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Send images or video frames to a YOLO model through anyserve.")
    parser.add_argument("--endpoint", help="default: ANYSERVE_ENDPOINT or http://127.0.0.1:50052")
    parser.add_argument("--model", default="yolo11n", help="the interface is yolo/<model>")
    parser.add_argument("--source", default="0", help="an image file, video file, RTSP URL, or camera index")
    parser.add_argument("--out", help="save annotated image or MP4 video (works headless in containers)")
    parser.add_argument("--frames", type=int, default=0, help="stop after this many frames; 0 plays the whole source")
    parser.add_argument("--budget-ms", type=int, default=1000, help="time one frame may take before it is dropped")
    parser.add_argument("--quality", type=int, default=85, help="JPEG encoding quality 1..100 (default: 85)")
    parser.add_argument("--conf", type=float, help="optional per-request confidence threshold")
    parser.add_argument("--labels", help="optional comma-separated class filter, e.g. person,car")
    parser.add_argument("--attr", action="append", default=[], metavar="KEY=VALUE", help="preferred worker attribute (repeatable)")
    parser.add_argument("--require", action="append", default=[], metavar="KEY=VALUE", help="required worker attribute (repeatable)")
    parser.add_argument("--show", action="store_true", help="draw boxes in an OpenCV window; q closes it")
    parser.add_argument("--json", action="store_true", help="print detections JSON (single image mode)")
    args = parser.parse_args()

    metadata: dict[str, str] = {}
    if args.conf is not None:
        metadata["conf"] = str(args.conf)
    if args.labels:
        metadata["labels"] = args.labels

    client = AnyserveClient(args.endpoint)
    options = ExecutionOptions(
        queue_timeout_ms=args.budget_ms,
        total_timeout_ms=args.budget_ms,
        max_attempts=1,
        metadata=metadata,
        preferred_attributes=parse_attrs(args.attr),
        required_attributes=parse_attrs(args.require),
    )
    interface = f"yolo/{args.model}"
    source_path = Path(args.source)

    # Single-image mode.
    if source_path.is_file() and source_path.suffix.lower() in IMAGE_EXTS:
        frame = cv2.imread(str(source_path))
        if frame is None:
            raise SystemExit(f"could not read image {args.source!r}")
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        if not ok:
            raise SystemExit("failed to encode JPEG")
        started = time.perf_counter()
        reply = json.loads(client.call(interface, buf.tobytes(), options=options))
        elapsed_ms = (time.perf_counter() - started) * 1000
        if args.out or args.show:
            draw_detections(frame, reply["detections"])
        if args.out:
            cv2.imwrite(args.out, frame)
        if args.json:
            print(json.dumps(reply, indent=2))
        else:
            summary = ", ".join(f"{d['label']}({d['confidence']:.2f})" for d in reply["detections"]) or "none"
            print(f"1 image in {elapsed_ms:.1f} ms (inference {reply['inference_ms']} ms): {len(reply['detections'])} objects [{summary}]")
            if args.out:
                print(f"saved annotated image to {args.out}")
        if args.show:
            cv2.imshow("yolo", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # Video / stream / camera mode.
    capture = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not capture.isOpened():
        raise SystemExit(f"could not open {args.source!r}")

    writer = None
    latencies: list[float] = []
    dropped = 0
    last_reply: dict | None = None
    started = time.perf_counter()
    try:
        while not args.frames or len(latencies) + dropped < args.frames:
            ok, frame = capture.read()
            if not ok:
                break
            enc_ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            if not enc_ok:
                dropped += 1
                continue
            sent = time.perf_counter()
            try:
                reply = json.loads(client.call(interface, buf.tobytes(), options=options))
            except RuntimeError:
                dropped += 1  # a late frame is worth nothing to live video
                continue
            latencies.append((time.perf_counter() - sent) * 1000)
            last_reply = reply
            if args.show or args.out:
                draw_detections(frame, reply["detections"])
            if args.out:
                if writer is None:
                    h, w = frame.shape[:2]
                    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))
                writer.write(frame)
            if args.show:
                cv2.imshow("yolo", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    if not latencies or last_reply is None:
        raise SystemExit(f"no frame was answered; {dropped} dropped")
    latencies.sort()
    print(
        f"{len(latencies)} frames at {len(latencies) / (time.perf_counter() - started):.1f} fps, {dropped} dropped, "
        f"latency p50 {latencies[len(latencies) // 2]:.1f} ms, max {latencies[-1]:.1f} ms; "
        f"last frame: {len(last_reply['detections'])} objects, inference {last_reply['inference_ms']} ms"
    )
    if args.out:
        print(f"saved annotated video to {args.out}")


if __name__ == "__main__":
    main()
