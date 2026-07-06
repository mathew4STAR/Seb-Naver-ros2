cmake_minimum_required(VERSION 3.5.1)
project(se2_grid_core)

set(CMAKE_CXX_STANDARD 14)
add_compile_options(-Wall -Wextra -Wpedantic)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

## Find dependencies
find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)

## Define Eigen addons (same as ROS1 — this file is ROS-independent)
include(cmake/${PROJECT_NAME}-extras.cmake)

###########
## Build ##
###########

add_library(${PROJECT_NAME}
   src/SE2Grid.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME}
  Eigen3::Eigen
)

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
  DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY cmake
  DESTINATION share/${PROJECT_NAME}
)

## Export for downstream packages
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_targets(export_${PROJECT_NAME})
ament_export_dependencies(Eigen3)

ament_package()
