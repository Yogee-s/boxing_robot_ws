from setuptools import setup
import os
from glob import glob

package_name = "boxbunny_gui"

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
    description="PySide6 GUI for BoxBunny drills and telemetry.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "gui_main = boxbunny_gui.gui_main:main",
        ],
    },
)
