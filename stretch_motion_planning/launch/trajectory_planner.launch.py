from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("driver_node", default_value="stretch_driver",
                              description="Driver node to read joint limits from ('' for none)"),
        Node(
            package="stretch_motion_planning",
            executable="planner_node.py",
            name="stretch_trajectory_planner",
            output="screen",
            parameters=[{"driver_node": LaunchConfiguration("driver_node")}],
        ),
    ])
