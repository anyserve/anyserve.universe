FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-vision-msgs \
    python3-colcon-common-extensions \
    python3-numpy \
    python3-pil \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages "anyserve-runtime>=0.7.0"

WORKDIR /ros2_ws
COPY . /ros2_ws/src/anyserve_ros2

RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash && colcon build --packages-select anyserve_ros2"

ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/jazzy/setup.bash && source /ros2_ws/install/setup.bash && exec \"$@\"", "--"]
CMD ["ros2", "run", "anyserve_ros2", "yolo_node"]
