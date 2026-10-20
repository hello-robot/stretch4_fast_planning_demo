from setuptools import setup

package_name = "stretch_dynamic_trajectory"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Hello Robot",
    maintainer_email="support@hello-robot.com",
    description="A FollowJointTrajectory server for Stretch 4 that slows down and stops short for obstacles.",
    license="Apache-2.0",
    tests_require=["pytest"],
)
