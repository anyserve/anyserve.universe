"""Pure numpy + Pillow conversions for ROS 2 sensor_msgs/msg/Image (no cv_bridge ABI lock-in)."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw
from sensor_msgs.msg import Image as RosImage


def ros_image_to_rgb(msg: RosImage) -> np.ndarray:
    """Decode a ROS 2 Image message into an HxWx3 uint8 RGB numpy array."""
    enc = msg.encoding.lower()
    h, w = int(msg.height), int(msg.width)
    raw = np.frombuffer(msg.data, dtype=np.uint8)

    if enc in ("rgb8", "bgr8"):
        row_bytes = int(msg.step) if msg.step else w * 3
        arr = raw.reshape((h, row_bytes))[:, : w * 3].reshape((h, w, 3))
        return arr[:, :, ::-1].copy() if enc == "bgr8" else arr.copy()
    if enc in ("rgba8", "bgra8"):
        row_bytes = int(msg.step) if msg.step else w * 4
        arr = raw.reshape((h, row_bytes))[:, : w * 4].reshape((h, w, 4))
        rgb = arr[:, :, :3]
        return rgb[:, :, ::-1].copy() if enc == "bgra8" else rgb.copy()
    if enc in ("mono8", "8uc1"):
        row_bytes = int(msg.step) if msg.step else w
        gray = raw.reshape((h, row_bytes))[:, :w]
        return np.stack([gray, gray, gray], axis=-1)
    raise ValueError(f"unsupported ROS image encoding {msg.encoding!r}")


def rgb_to_ros_image(rgb: np.ndarray, header) -> RosImage:
    """Encode an HxWx3 uint8 RGB numpy array as a ROS 2 rgb8 Image message."""
    contig = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = contig.shape[:2]
    msg = RosImage()
    msg.header = header
    msg.height = h
    msg.width = w
    msg.encoding = "rgb8"
    msg.is_bigendian = 0
    msg.step = w * 3
    msg.data = contig.tobytes()
    return msg


def rgb_to_jpeg(rgb: np.ndarray, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    PILImage.fromarray(rgb, mode="RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def draw_boxes_rgb(rgb: np.ndarray, detections: list[dict]) -> np.ndarray:
    pil = PILImage.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(pil)
    w, h = pil.size
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        ix1 = max(0, int(round(x1)))
        iy1 = max(0, int(round(y1)))
        ix2 = min(w - 1, int(round(x2)))
        iy2 = min(h - 1, int(round(y2)))
        draw.rectangle([ix1, iy1, ix2, iy2], outline=(0, 220, 80), width=2)
        draw.text((ix1 + 4, max(0, iy1 - 12)), f"{det['label']} {det['confidence']:.2f}", fill=(0, 220, 80))
    return np.asarray(pil, dtype=np.uint8)
