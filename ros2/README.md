# ROS 2 Nodes (`anyserve_ros2`)

A ROS 2 (`ament_python`) package that offloads real-time robot perception and
vision-language-action (VLA) policy inference to remote GPU workers over
[AnyServe](https://github.com/anyserve/anyserve). Robots behind cellular NAT or
private LANs dial **out** over gRPC or `wss://` with zero open inbound ports.

## Included nodes

| Executable | Subscribes | Calls (AnyServe) | Publishes |
|---|---|---|---|
| `yolo_node` | `/camera/image_raw` (`sensor_msgs/Image`) | `yolo/<model>` | `/detections` (`vision_msgs/Detection2DArray`), `/detections_json` (`std_msgs/String`), `/detections/image` (`sensor_msgs/Image`) |
| `vla_node` | camera `sensor_msgs/Image` topics, `/joint_states` (`sensor_msgs/JointState`), `/vla/instruction` (`std_msgs/String`) | `vla/<model>` | `/joint_commands` (`sensor_msgs/JointState`, 30 Hz), `/vla/diagnostics` (`std_msgs/String`) |
| `demo_publisher` | — | — | `/camera/image_raw`, `/joint_states`, `/vla/instruction` for testing without hardware |

### Robotics-oriented design

- **Latest-frame-wins**: callbacks never block the ROS 2 executor; a dedicated
  inference thread grabs the newest frame and drops stale frames under load.
- **Hard real-time budgets**: calls set `queue_timeout_ms = total_timeout_ms = budget_ms`
  and `max_attempts = 1` so late perception or policy responses are dropped
  instead of piling up a stale queue.
- **Zero `cv_bridge` ABI fragility**: `image_codec.py` decodes `rgb8`, `bgr8`,
  `rgba8`, `bgra8`, and `mono8` directly with NumPy and Pillow.
- **Edge-to-cloud spillover**: configure `preferred_attributes` (e.g.
  `site=edge`) and `spillover_after_ms` to prefer a local GPU and spill over to
  a remote worker automatically.

## Build with `colcon`

Tested on ROS 2 Jazzy (Ubuntu 24.04, Python 3.12):

```bash
pip install anyserve-runtime
mkdir -p ~/ros2_ws/src
cp -r ros2 ~/ros2_ws/src/anyserve_ros2
cd ~/ros2_ws
colcon build --packages-select anyserve_ros2
source install/setup.bash
```

## Run

Start the AnyServe control plane and `yolo/worker.py` (and/or `vla/worker.py`)
on a GPU machine, then launch the ROS 2 nodes on the robot:

```bash
export ANYSERVE_ENDPOINT="http://<control-plane>:50052"
export ANYSERVE_TOKEN="<client-token>"

# Perception offload (YOLO)
ros2 run anyserve_ros2 yolo_node --ros-args \
  -p interface:=yolo/yolo11n \
  -p image_topic:=/camera/image_raw \
  -p budget_ms:=500 \
  -p preferred_attributes:="site=edge"

# Closed-loop VLA control (SmolVLA)
ros2 run anyserve_ros2 vla_node --ros-args \
  -p interface:=vla/lerobot/smolvla_base \
  -p cameras:="camera1=/camera/top/image_raw,camera2=/camera/wrist/image_raw" \
  -p control_hz:=30.0 \
  -p execute_steps:=25 \
  -p budget_ms:=1500
```

## Run in Docker

```bash
docker build -t anyserve-ros2 ./ros2
docker run --rm --network host \
  -e ANYSERVE_ENDPOINT=http://127.0.0.1:50052 \
  -e ANYSERVE_TOKEN="<client-token>" \
  anyserve-ros2 ros2 run anyserve_ros2 yolo_node
```

## Parameters

### `yolo_node`

| Parameter | Default | Purpose |
|---|---|---|
| `endpoint` | `""` (uses `ANYSERVE_ENDPOINT`) | Control plane URL (`http(s)://` or `ws(s)://`) |
| `interface` | `yolo/yolo11n` | Target AnyServe interface |
| `image_topic` | `/camera/image_raw` | Input `sensor_msgs/Image` topic (`BEST_EFFORT`, depth 1) |
| `detections_topic` | `/detections` | Output `vision_msgs/Detection2DArray` topic |
| `detections_json_topic` | `/detections_json` | Output JSON `std_msgs/String` topic |
| `annotated_topic` | `/detections/image` | Annotated `rgb8` image topic for RViz2 / Foxglove |
| `publish_annotated` | `true` | Enable/disable drawing and publishing annotated frames |
| `budget_ms` | `500` | Queue + total deadline before dropping a frame |
| `min_confidence` | `0.25` | Per-request confidence threshold |
| `labels` | `""` | Optional comma-separated class filter (e.g. `person,car`) |
| `preferred_attributes` / `required_attributes` | `""` | Comma-separated `key=value` worker placement attributes |
| `spillover_after_ms` | `0` | Wait for preferred workers before spilling over |

### `vla_node`

| Parameter | Default | Purpose |
|---|---|---|
| `interface` | `vla/lerobot/smolvla_base` | Target VLA worker interface |
| `cameras` | `camera1=/camera/image_raw` | Comma-separated `policy_cam=/ros/topic` mappings |
| `joint_states_topic` | `/joint_states` | Current robot joint positions |
| `instruction_topic` | `/vla/instruction` | Live task instruction topic |
| `joint_commands_topic` | `/joint_commands` | Output `sensor_msgs/JointState` action targets |
| `control_hz` | `30.0` | Control loop timer rate |
| `execute_steps` | `25` | Control ticks between policy chunk refreshes |
| `budget_ms` | `1500` | Policy call deadline |

## Container verification (ROS 2 Jazzy + Tesla T4)

Verified inside `ros:jazzy-ros-base` (`colcon build --packages-select anyserve_ros2`)
talking to authenticated AnyServe v0.7.0 GPU workers over a Docker bridge network:

| Node | Remote worker | Observed round-trip | GPU inference | Output verified |
|---|---|---|---|---|
| `yolo_node` | `yolo/yolo11n` (810×1080 JPEG) | **22.6 ms** | 14.7 ms | 5 `vision_msgs/Detection2DArray` objects + `/detections_json` + annotated `/detections/image` |
| `vla_node` | `vla/lerobot/smolvla_base` | **599.6 ms** | 595.2 ms | 50-step action chunks published as 30 Hz `/joint_commands` (`sensor_msgs/JointState`) + `/vla/diagnostics` |
