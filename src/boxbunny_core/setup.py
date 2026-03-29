import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'boxbunny_core'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # ament index
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        # package manifest
        ('share/' + package_name, ['package.xml']),
        # config files
        ('share/' + package_name + '/config', glob('config/*')),
        # launch files
        ('share/' + package_name + '/launch', glob('launch/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='boxbunny',
    maintainer_email='boxbunny@todo.com',
    description='Core processing package for the BoxBunny boxing training robot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cv_node = boxbunny_core.cv_node:main',
            'imu_node = boxbunny_core.imu_node:main',
            'robot_node = boxbunny_core.robot_node:main',
            'punch_processor = boxbunny_core.punch_processor:main',
            'session_manager = boxbunny_core.session_manager:main',
            'drill_manager = boxbunny_core.drill_manager:main',
            'sparring_engine = boxbunny_core.sparring_engine:main',
            'analytics_node = boxbunny_core.analytics_node:main',
            'llm_node = boxbunny_core.llm_node:main',
        ],
    },
)
