"""ROS 2 perception offload node: subscribes to camera frames and calls AnyServe YOLO workers.

Design choices for robotics:
- SensorData (BEST_EFFORT, depth=1) subscription compatible with both RELIABLE
  and BEST_EFFORT camera drivers.
- Latest-frame-wins background thread: never blocks the ROS 2 executor and never
  builds a stale frame queue when network latency spikes.
- Hard per-frame deadline (``budget_ms``) with ``max_attempts=1``: a late
  detection is dropped so the next fresh frame runs immediately.
"""
from __future__ import annotations

import json
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import String

from anyserve import AnyserveClient, ExecutionOptions

from .image_codec import draw_boxes_rgb, rgb_to_jpeg, rgb_to_ros_image, ros_image_to_rgb

try:
    from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

    HAS_VISION_MSGS = True
except ImportError:
    HAS_VISION_MSGS = False


def _parse_kv(raw: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for item in (raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"invalid attribute {item!r}: expected key=value")
        k, v = item.split("=", 1)
        attrs[k.strip()] = v.strip()
    return attrs


class AnyServeYoloNode(Node):
    def __init__(self) -> None:
        super().__init__("anyserve_yolo_node")
        self.declare_parameter("endpoint", "")
        self.declare_parameter("token", "")
        self.declare_parameter("namespace", "")
        self.declare_parameter("interface", "yolo/yolo11n")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("detections_topic", "/detections")
        self.declare_parameter("detections_json_topic", "/detections_json")
        self.declare_parameter("annotated_topic", "/detections/image")
        self.declare_parameter("publish_annotated", True)
        self.declare_parameter("budget_ms", 500)
        self.declare_parameter("jpeg_quality", 85)
        self.declare_parameter("min_confidence", 0.25)
        self.declare_parameter("labels", "")
        self.declare_parameter("preferred_attributes", "")
        self.declare_parameter("required_attributes", "")
        self.declare_parameter("spillover_after_ms", 0)

        endpoint = str(self.get_parameter("endpoint").value) or None
        token = str(self.get_parameter("token").value) or None
        namespace = str(self.get_parameter("namespace").value) or None
        self._interface = str(self.get_parameter("interface").value)
        self._budget_ms = int(self.get_parameter("budget_ms").value)
        self._jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self._publish_annotated = bool(self.get_parameter("publish_annotated").value)

        metadata = {"conf": str(self.get_parameter("min_confidence").value)}
        labels = str(self.get_parameter("labels").value).strip()
        if labels:
            metadata["labels"] = labels

        self._client = AnyserveClient(endpoint, token=token, namespace=namespace)
        self._options = ExecutionOptions(
            queue_timeout_ms=self._budget_ms,
            total_timeout_ms=self._budget_ms,
            max_attempts=1,
            metadata=metadata,
            preferred_attributes=_parse_kv(str(self.get_parameter("preferred_attributes").value)),
            required_attributes=_parse_kv(str(self.get_parameter("required_attributes").value)),
            spillover_after_ms=int(self.get_parameter("spillover_after_ms").value),
        )

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        image_topic = str(self.get_parameter("image_topic").value)
        self.create_subscription(RosImage, image_topic, self._on_image, sensor_qos)

        self._json_pub = self.create_publisher(
            String, str(self.get_parameter("detections_json_topic").value), 10
        )
        self._vision_pub = (
            self.create_publisher(Detection2DArray, str(self.get_parameter("detections_topic").value), 10)
            if HAS_VISION_MSGS
            else None
        )
        self._annotated_pub = (
            self.create_publisher(RosImage, str(self.get_parameter("annotated_topic").value), 10)
            if self._publish_annotated
            else None
        )

        self._lock = threading.Lock()
        self._latest: RosImage | None = None
        self._new_frame = threading.Event()
        self._stop = threading.Event()
        self._stats = {"received": 0, "processed": 0, "dropped_busy": 0, "dropped_timeout": 0}
        self._worker_thread = threading.Thread(target=self._loop, name="anyserve-yolo-loop", daemon=True)
        self._worker_thread.start()

        self.get_logger().info(
            f"AnyServe YOLO node ready: {image_topic} -> {self._interface} "
            f"(budget={self._budget_ms}ms, vision_msgs={HAS_VISION_MSGS})"
        )

    def _on_image(self, msg: RosImage) -> None:
        with self._lock:
            if self._latest is not None:
                self._stats["dropped_busy"] += 1
            self._latest = msg
            self._stats["received"] += 1
        self._new_frame.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self._new_frame.wait(timeout=0.2):
                continue
            self._new_frame.clear()
            with self._lock:
                msg = self._latest
                self._latest = None
            if msg is None:
                continue

            try:
                rgb = ros_image_to_rgb(msg)
                jpeg = rgb_to_jpeg(rgb, quality=self._jpeg_quality)
            except Exception as exc:
                self.get_logger().warning(f"failed to encode camera frame: {exc}")
                continue

            sent = time.perf_counter()
            try:
                reply = json.loads(self._client.call(self._interface, jpeg, options=self._options))
            except RuntimeError as exc:
                self._stats["dropped_timeout"] += 1
                self.get_logger().warning(f"frame dropped ({self._budget_ms}ms budget): {str(exc)[:120]}")
                continue

            rtt_ms = round((time.perf_counter() - sent) * 1000, 1)
            detections = reply.get("detections", [])
            self._stats["processed"] += 1

            payload = {
                "frame_id": msg.header.frame_id,
                "stamp": {"sec": msg.header.stamp.sec, "nanosec": msg.header.stamp.nanosec},
                "interface": self._interface,
                "detections": detections,
                "inference_ms": reply.get("inference_ms"),
                "round_trip_ms": rtt_ms,
                "stats": dict(self._stats),
            }
            out_json = String()
            out_json.data = json.dumps(payload)
            self._json_pub.publish(out_json)

            if self._vision_pub is not None:
                self._vision_pub.publish(self._to_detection_array(msg.header, detections))

            if self._annotated_pub is not None:
                annotated = draw_boxes_rgb(rgb, detections)
                self._annotated_pub.publish(rgb_to_ros_image(annotated, msg.header))

    @staticmethod
    def _to_detection_array(header, detections: list[dict]):
        arr = Detection2DArray()
        arr.header = header
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cx = float(x1 + x2) / 2.0
            cy = float(y1 + y2) / 2.0
            sx = max(0.0, float(x2 - x1))
            sy = max(0.0, float(y2 - y1))
            item = Detection2D()
            item.header = header
            if hasattr(item.bbox.center, "position"):
                item.bbox.center.position.x = cx
                item.bbox.center.position.y = cy
            else:
                item.bbox.center.x = cx
                item.bbox.center.y = cy
            item.bbox.size_x = sx
            item.bbox.size_y = sy
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(det["label"])
            hyp.hypothesis.score = float(det["confidence"])
            item.results.append(hyp)
            arr.detections.append(item)
        return arr

    def destroy_node(self) -> bool:
        self._stop.set()
        self._new_frame.set()
        self._worker_thread.join(timeout=2.0)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AnyServeYoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
