import os
from glob import glob

from setuptools import find_packages, setup


package_name = "embodied_bringup"


setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml"],
        ),
        (
            os.path.join(
                "share",
                package_name,
                "launch",
            ),
            glob("launch/*.launch.py"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="liyu",
    maintainer_email="18649501318@163.com",
    description=(
        "Launch and bringup package for the "
        "embodied robotic arm system"
    ),
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [],
    },
)
