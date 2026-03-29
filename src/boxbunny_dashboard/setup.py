from glob import glob

from setuptools import find_packages, setup

package_name = "boxbunny_dashboard"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        # ament index
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        # package manifest
        ("share/" + package_name, ["package.xml"]),
        # static assets (SPA build output) — include root files + assets subdir
        ("share/" + package_name + "/static/dist",
         [f for f in glob("static/dist/*") if not f.endswith("/assets") and "." in f.split("/")[-1]]),
        ("share/" + package_name + "/static/dist/assets",
         glob("static/dist/assets/*")),
        # SQL data files
        ("share/" + package_name + "/data",
         [f for f in glob("data/*") if "." in f.split("/")[-1]]),
    ],
    install_requires=[
        "setuptools",
        "fastapi",
        "uvicorn",
        "pydantic",
        "bcrypt",
    ],
    zip_safe=True,
    maintainer="boxbunny",
    maintainer_email="boxbunny@todo.com",
    description="Mobile-first web dashboard for BoxBunny boxing training robot",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "dashboard_server = boxbunny_dashboard.server:main",
        ],
    },
)
