from setuptools import setup
import os
from glob import glob

package_name = "boxbunny_vision"

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
    description="RealSense glove tracking and punch prediction.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "realsense_glove_tracker = boxbunny_vision.realsense_glove_tracker:main",
            "action_predictor = boxbunny_vision.action_predictor_node:main",
            "simple_camera_node = boxbunny_vision.simple_camera_node:main",
        ],
    },
)
