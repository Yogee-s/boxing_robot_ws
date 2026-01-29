from setuptools import setup
import os
from glob import glob

package_name = "boxbunny_drills"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="BoxBunny",
    maintainer_email="you@example.com",
    description="Reaction drill manager and logging.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "reaction_drill_manager = boxbunny_drills.reaction_drill_manager:main",
            "shadow_sparring_drill = boxbunny_drills.shadow_sparring_drill:main",
            "defence_drill = boxbunny_drills.defence_drill:main",
        ],
    },
)
