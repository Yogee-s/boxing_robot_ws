from setuptools import setup
import os
from glob import glob

package_name = "boxbunny_imu"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="BoxBunny",
    maintainer_email="you@example.com",
    description="MPU6050 IMU integration and optional punch classification.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "mpu6050_node = boxbunny_imu.mpu6050_node:main",
            "imu_punch_classifier = boxbunny_imu.imu_punch_classifier:main",
        ],
    },
)
