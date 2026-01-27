from setuptools import setup
import os
from glob import glob

package_name = "boxbunny_msgs"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "msg"), glob("msg/*.msg")),
        (os.path.join("share", package_name, "srv"), glob("srv/*.srv")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="BoxBunny",
    maintainer_email="you@example.com",
    description="Custom messages and services for BoxBunny.",
    license="MIT",
)
