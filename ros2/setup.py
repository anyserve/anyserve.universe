from setuptools import find_packages, setup

package_name = "anyserve_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools", "anyserve-runtime>=0.7.0", "numpy", "Pillow"],
    zip_safe=True,
    maintainer="anyserve",
    maintainer_email="arcosx@outlook.com",
    description="ROS 2 nodes for offloading perception and VLA policies over AnyServe.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "yolo_node = anyserve_ros2.yolo_node:main",
            "vla_node = anyserve_ros2.vla_node:main",
            "demo_publisher = anyserve_ros2.demo_publisher:main",
        ],
    },
)
