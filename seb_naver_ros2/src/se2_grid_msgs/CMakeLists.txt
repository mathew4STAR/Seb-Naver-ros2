cmake_minimum_required(VERSION 3.5.1)
project(se2_grid_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

# Generate ROS2 message interfaces
# .msg file syntax is identical between ROS1 and ROS2
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SE2GridInfo.msg"
  "msg/SE2Grid.msg"
  DEPENDENCIES
    std_msgs
    geometry_msgs
    builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
