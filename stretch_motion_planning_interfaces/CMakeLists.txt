cmake_minimum_required(VERSION 3.8)
project(stretch_motion_planning_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/PlanJointTrajectory.srv"
  "srv/CheckMotion.srv"
  DEPENDENCIES builtin_interfaces geometry_msgs sensor_msgs shape_msgs trajectory_msgs
)

ament_package()
