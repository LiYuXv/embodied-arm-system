from setuptools import find_packages, setup


package_name = "embodied_language"


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
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="liyu",
    maintainer_email="18649501318@163.com",
    description=(
        "Natural-language command parsing for the "
        "embodied robotic arm system"
    ),
    license="TODO: License declaration",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            (
                "language_node = "
                "embodied_language.language_node:main"
            ),
        ],
    },
)
