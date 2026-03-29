import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'boxbunny_gui'

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
        # assets — sounds
        ('share/' + package_name + '/assets/sounds', glob('assets/sounds/*')),
        # assets — icons
        ('share/' + package_name + '/assets/icons', glob('assets/icons/*')),
        # assets — fonts
        ('share/' + package_name + '/assets/fonts', glob('assets/fonts/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='boxbunny',
    maintainer_email='boxbunny@todo.com',
    description='PySide6 touchscreen GUI for the BoxBunny boxing training robot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gui_main = boxbunny_gui.app:main',
        ],
    },
)
