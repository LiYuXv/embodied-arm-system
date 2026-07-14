from glob import glob

from setuptools import setup


package_name = "gazebo_classic_sim"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/gazebo_classic_sim"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/worlds", glob("worlds/*.world")),
        (f"share/{package_name}/scripts", glob("scripts/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="liyu",
    maintainer_email="liyu@example.com",
    description="Minimal Gazebo Classic validation for the EDULITE A3.",
    license="Apache-2.0",
)
