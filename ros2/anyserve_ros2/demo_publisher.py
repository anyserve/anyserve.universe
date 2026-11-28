"""Publish sample camera images and joint states for testing anyserve_ros2 without hardware."""
from __future__ import annotations

import math

import numpy as np
import rclpy
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from .image_codec import rgb_to_ros_image


class DemoRobotPublisher(Node):
    def __init__(self) -> None:
        super().__init__("anyserve_demo_publisher")
        self.declare_parameter("image_path", "")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("instruction_topic", "/vla/instruction")
        self.declare_parameter("instruction", "Pick up the block and place it in the bin.")
        self.declare_parameter("fps", 5.0)
        self.declare_parameter("joints", 6)

        image_path = str(self.get_parameter("image_path").value)
        if image_path:
            self._frame = np.asarray(PILImage.open(image_path).convert("RGB"), dtype=np.uint8)
        else:
            # Generate a 640x480 synthetic test image when no file is provided.
            y, x = np.mgrid[0:480, 0:640]
            self._frame = np.stack(
                [
                    (x * 255 // 640).astype(np.uint8),
                    (y * 255 // 480).astype(np.uint8),
                    np.full((480, 640), 120, dtype=np.uint8),
                ],
                axis=-1,
            )

        self._img_pub = self.create_publisher(RosImage, str(self.get_parameter("image_topic").value), 10)
        self._joint_pub = self.create_publisher(JointState, str(self.get_parameter("joint_states_topic").value), 10)
        self._instr_pub = self.create_publisher(String, str(self.get_parameter("instruction_topic").value), 10)
        self._joints = int(self.get_parameter("joints").value)
        self._instruction = str(self.get_parameter("instruction").value)
        self._seq = 0

        fps = max(0.5, float(self.get_parameter("fps").value))
        self.create_timer(1.0 / fps, self._tick)
        self.get_logger().info(
            f"Publishing demo camera ({self._frame.shape[1]}x{self._frame.shape[0]}) and {self._joints} joints at {fps} Hz"
        )

    def _tick(self) -> None:
        now = self.get_clock().now().to_msg()
        img = RosImage()
        img.header.stamp = now
        img.header.frame_id = "camera_optical_frame"
        self._img_pub.publish(rgb_to_ros_image(self._frame, img.header))

        js = JointState()
        js.header.stamp = now
        js.name = [f"joint_{i}" for i in range(self._joints)]
        t = self._seq * 0.1
        js.position = [round(0.25 * math.sin(t + i * 0.5), 4) for i in range(self._joints)]
        self._joint_pub.publish(js)

        if self._seq == 0:
            s = String()
            s.data = self._instruction
            self._instr_pub.publish(s)
        self._seq += 1


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DemoRobotPublisher()
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
