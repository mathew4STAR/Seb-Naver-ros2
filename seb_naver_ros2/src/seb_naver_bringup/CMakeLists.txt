cmake_minimum_required(VERSION 3.8)
project(seb_naver_bringup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(se2_grid_core REQUIRED)
find_package(se2_grid_ros REQUIRED)
find_package(se2_grid_msgs REQUIRED)

add_executable(se2_to_occupancy_node src/se2_to_occupancy_node.cpp)
ament_target_dependencies(se2_to_occupancy_node
  rclcpp
  nav_msgs
  se2_grid_core
  se2_grid_ros
  se2_grid_msgs
)

install(TARGETS
  se2_to_occupancy_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch params
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
