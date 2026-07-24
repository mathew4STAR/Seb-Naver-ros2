cmake_minimum_required(VERSION 3.5.1)
project(se2_grid_ros)

set(CMAKE_CXX_STANDARD 14)
add_compile_options(-Wall -Wextra -Wpedantic)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

## Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(se2_grid_core REQUIRED)
find_package(se2_grid_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(Eigen3 REQUIRED)

## NOTE: rosbag, tf, cv_bridge dropped — not needed for core conversion functionality.
## rosbag → saveToBag/loadFromBag functions removed (use ros2 bag CLI instead)
## tf → not used in this package's source code
## cv_bridge → not used in this package's source code

###########
## Build ##
###########

add_library(${PROJECT_NAME} SHARED
  src/SE2GridRosConverter.cpp
  src/SE2GridMsgHelpers.cpp
)

add_executable(se2_visualizer_node src/se2_visualizer_node.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIR}
)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  se2_grid_core
  se2_grid_msgs
  sensor_msgs
  nav_msgs
  std_msgs
  geometry_msgs
  builtin_interfaces
)

ament_target_dependencies(se2_visualizer_node
  rclcpp
  se2_grid_core
  se2_grid_msgs
  nav_msgs
  std_msgs
)
target_link_libraries(se2_visualizer_node ${PROJECT_NAME})

# Ensure message headers are generated before we build
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

#############
## Install ##
#############

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  TARGETS se2_visualizer_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include
)

## Export for downstream packages
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_targets(export_${PROJECT_NAME})
ament_export_dependencies(
  rclcpp se2_grid_core se2_grid_msgs
  sensor_msgs nav_msgs std_msgs geometry_msgs builtin_interfaces
)

ament_package()
