"""ROS 2 closed-loop VLA policy node: streams robot observations to remote AnyServe VLA workers.

Subscribes to camera images (``sensor_msgs/Image``), ``/joint_states``
(``sensor_msgs/JointState``), and ``/vla/instruction`` (``std_msgs/String``).
A deterministic control timer (default 30 Hz) publishes ``/joint_commands`` from
the active action chunk while a background thread refreshes chunks asynchronously
every ``execute_steps`` ticks without blocking the control loop.
"""
from __future__ import annotations

import base64
import json
import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from anyserve import AnyserveClient, ExecutionOptions

from .image_codec import rgb_to_jpeg, ros_image_to_rgb


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


class AnyServeVlaNode(Node):
    def __init__(self) -> None:
        super().__init__("anyserve_vla_node")
        self.declare_parameter("endpoint", "")
        self.declare_parameter("token", "")
        self.declare_parameter("namespace", "")
        self.declare_parameter("interface", "vla/lerobot/smolvla_base")
        self.declare_parameter("cameras", "camera1=/camera/image_raw")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("instruction_topic", "/vla/instruction")
        self.declare_parameter("joint_commands_topic", "/joint_commands")
        self.declare_parameter("diagnostics_topic", "/vla/diagnostics")
        self.declare_parameter("default_instruction", "Pick up the object and place it in the bin.")
        self.declare_parameter("control_hz", 30.0)
        self.declare_parameter("execute_steps", 25)
        self.declare_parameter("budget_ms", 1500)
        self.declare_parameter("jpeg_quality", 85)
        self.declare_parameter("preferred_attributes", "")
        self.declare_parameter("required_attributes", "")

        endpoint = str(self.get_parameter("endpoint").value) or None
        token = str(self.get_parameter("token").value) or None
        namespace = str(self.get_parameter("namespace").value) or None
        self._interface = str(self.get_parameter("interface").value)
        self._control_hz = float(self.get_parameter("control_hz").value)
        self._execute_steps = max(1, int(self.get_parameter("execute_steps").value))
        self._budget_ms = int(self.get_parameter("budget_ms").value)
        self._jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self._camera_map = _parse_kv(str(self.get_parameter("cameras").value))

        self._client = AnyserveClient(endpoint, token=token, namespace=namespace)
        self._options = ExecutionOptions(
            queue_timeout_ms=self._budget_ms,
            total_timeout_ms=self._budget_ms,
            max_attempts=1,
            preferred_attributes=_parse_kv(str(self.get_parameter("preferred_attributes").value)),
            required_attributes=_parse_kv(str(self.get_parameter("required_attributes").value)),
        )

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        for cam_name, topic in self._camera_map.items():
            self.create_subscription(
                RosImage,
                topic,
                lambda msg, name=cam_name: self._on_camera(name, msg),
                sensor_qos,
            )

        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_states_topic").value),
            self._on_joint_state,
            sensor_qos,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("instruction_topic").value),
            self._on_instruction,
            10,
        )

        self._cmd_pub = self.create_publisher(
            JointState, str(self.get_parameter("joint_commands_topic").value), 10
        )
        self._diag_pub = self.create_publisher(
            String, str(self.get_parameter("diagnostics_topic").value), 10
        )

        self._lock = threading.Lock()
        self._latest_images: dict[str, RosImage] = {}
        self._latest_joints: list[float] | None = None
        self._joint_names: list[str] = []
        self._instruction = str(self.get_parameter("default_instruction").value)
        self._chunk: list[list[float]] = []
        self._chunk_index = 0
        self._ticks_since_request = self._execute_steps
        self._in_flight = False
        self._stats = {"calls": 0, "dropped": 0, "commands_published": 0, "last_rtt_ms": 0.0, "last_inference_ms": 0.0}

        self._request_event = threading.Event()
        self._stop = threading.Event()
        self._policy_thread = threading.Thread(target=self._policy_loop, name="anyserve-vla-loop", daemon=True)
        self._policy_thread.start()
        self._timer = self.create_timer(1.0 / max(1.0, self._control_hz), self._on_control_tick)

        self.get_logger().info(
            f"AnyServe VLA node ready: interface={self._interface}, cameras={self._camera_map}, "
            f"control_hz={self._control_hz}, execute_steps={self._execute_steps}"
        )

    def _on_camera(self, name: str, msg: RosImage) -> None:
        with self._lock:
            self._latest_images[name] = msg

    def _on_joint_state(self, msg: JointState) -> None:
        with self._lock:
            self._latest_joints = [float(v) for v in msg.position]
            if msg.name:
                self._joint_names = list(msg.name)

    def _on_instruction(self, msg: String) -> None:
        with self._lock:
            self._instruction = msg.data

    def _on_control_tick(self) -> None:
        with self._lock:
            self._ticks_since_request += 1
            if (
                self._latest_joints is not None
                and not self._in_flight
                and (self._ticks_since_request >= self._execute_steps or self._chunk_index >= len(self._chunk))
            ):
                self._in_flight = True
                self._ticks_since_request = 0
                self._request_event.set()

            if self._chunk_index >= len(self._chunk):
                return
            action = self._chunk[self._chunk_index]
            self._chunk_index += 1
            names = list(self._joint_names)
            self._stats["commands_published"] += 1

        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.name = names if len(names) == len(action) else [f"joint_{i}" for i in range(len(action))]
        cmd.position = [float(v) for v in action]
        self._cmd_pub.publish(cmd)

    def _policy_loop(self) -> None:
        while not self._stop.is_set():
            if not self._request_event.wait(timeout=0.2):
                continue
            self._request_event.clear()

            with self._lock:
                state = list(self._latest_joints) if self._latest_joints is not None else None
                images_snapshot = dict(self._latest_images)
                instruction = self._instruction

            if state is None:
                with self._lock:
                    self._in_flight = False
                continue

            encoded_images: dict[str, str] = {}
            for name, img_msg in images_snapshot.items():
                try:
                    rgb = ros_image_to_rgb(img_msg)
                    encoded_images[name] = base64.b64encode(
                        rgb_to_jpeg(rgb, quality=self._jpeg_quality)
                    ).decode()
                except Exception as exc:
                    self.get_logger().warning(f"failed to encode camera {name!r}: {exc}")

            payload = json.dumps(
                {"instruction": instruction, "state": state, "images": encoded_images}
            ).encode()

            sent = time.perf_counter()
            try:
                reply = json.loads(self._client.call(self._interface, payload, options=self._options))
                elapsed = time.perf_counter() - sent
                actions = reply.get("actions") or []
                skip = min(max(0, len(actions) - 1), math.floor(elapsed * self._control_hz))
                with self._lock:
                    if actions:
                        self._chunk = actions
                        self._chunk_index = skip
                    self._stats["calls"] += 1
                    self._stats["last_rtt_ms"] = round(elapsed * 1000, 1)
                    self._stats["last_inference_ms"] = float(reply.get("inference_ms", 0.0))
                    diag = {
                        **self._stats,
                        "chunk_size": len(actions),
                        "chunk_offset": skip,
                        "instruction": instruction,
                    }
            except RuntimeError as exc:
                with self._lock:
                    self._stats["dropped"] += 1
                    diag = {**self._stats, "error": str(exc)[:120]}
                self.get_logger().warning(f"VLA policy call missed budget ({self._budget_ms}ms): {str(exc)[:120]}")
            finally:
                with self._lock:
                    self._in_flight = False

            msg = String()
            msg.data = json.dumps(diag)
            self._diag_pub.publish(msg)

    def destroy_node(self) -> bool:
        self._stop.set()
        self._request_event.set()
        self._policy_thread.join(timeout=2.0)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AnyServeVlaNode()
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
