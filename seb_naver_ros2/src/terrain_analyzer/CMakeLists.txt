cmake_minimum_required(VERSION 3.5.1)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()
if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()

set(ENABLE_CUDA true)

if(ENABLE_CUDA)
  set(CMAKE_CUDA_ARCHITECTURES "87") # Jetson Orin NX
  project(terrain_analyzer CXX CUDA)
  add_definitions(-DENABLE_CUDA)
else(ENABLE_CUDA)
  project(terrain_analyzer)
endif(ENABLE_CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

## Find dependencies — ROS2 ament
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(se2_grid_core REQUIRED)
find_package(se2_grid_ros REQUIRED)
find_package(se2_grid_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(pluginlib REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)

## Non-ROS dependencies
find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters kdtree io)
find_package(yaml-cpp REQUIRED)

include_directories(include)
include_directories(SYSTEM
    ${Boost_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIRS}
)

if(ENABLE_CUDA)

  set(CUDA_SEPARABLE_COMPILATION ON)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -use_fast_math;)
  set(CUDA_NVCC_FLAGS
        # Jetson Orin NX
        -gencode arch=compute_87,code=sm_87;
        # RTX 2060
        # -gencode arch=compute_75,code=sm_75;
  )
  set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

  add_library(${PROJECT_NAME}_library
    src/TerrainAnalyzer.cpp
  )

  add_library(cuda_computer src/TerrainBank.cu)
  target_link_libraries(cuda_computer
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
  )
  ament_target_dependencies(cuda_computer
    cv_bridge
    sensor_msgs
    std_msgs
  )

  target_link_libraries(${PROJECT_NAME}_library
    cuda_computer
    ${Boost_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${PCL_LIBRARIES}
    yaml-cpp
  )

  ament_target_dependencies(${PROJECT_NAME}_library
    rclcpp
    se2_grid_core
    se2_grid_ros
    se2_grid_msgs
    sensor_msgs
    std_msgs
    geometry_msgs
    visualization_msgs
    nav_msgs
    message_filters
    pcl_conversions
    cv_bridge
    pluginlib
    tf2_ros
    tf2_eigen
  )

else(ENABLE_CUDA)

  add_library(${PROJECT_NAME}_library
    src/TerrainAnalyzer.cpp
  )

  target_link_libraries(${PROJECT_NAME}_library
    ${OpenCV_LIBRARIES}
    ${PCL_LIBRARIES}
    yaml-cpp
    pthread
  )

  ament_target_dependencies(${PROJECT_NAME}_library
    rclcpp
    se2_grid_core
    se2_grid_ros
    se2_grid_msgs
    sensor_msgs
    std_msgs
    geometry_msgs
    visualization_msgs
    nav_msgs
    message_filters
    pcl_conversions
    cv_bridge
    pluginlib
    tf2_ros
    tf2_eigen
  )

endif(ENABLE_CUDA)

##############
# Executable #
##############

add_executable(${PROJECT_NAME}_node
  src/terrain_analyzer_node.cpp
)

target_link_libraries(${PROJECT_NAME}_node
  ${PROJECT_NAME}_library
)

ament_target_dependencies(${PROJECT_NAME}_node
  rclcpp
)

# plugin
add_library(terrain_analyzer_post_processor_plugin SHARED
  src/post_processors.cpp
)

target_include_directories(terrain_analyzer_post_processor_plugin
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(terrain_analyzer_post_processor_plugin SYSTEM
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIRS}
)

target_link_libraries(terrain_analyzer_post_processor_plugin
  ${OpenCV_LIBRARIES}
  yaml-cpp
)

ament_target_dependencies(terrain_analyzer_post_processor_plugin
  rclcpp
  se2_grid_core
  se2_grid_ros
  cv_bridge
  sensor_msgs
  pluginlib
)

#############
## Install ##
#############

install(
  TARGETS
    ${PROJECT_NAME}_node
  DESTINATION
    lib/${PROJECT_NAME}
)

if(ENABLE_CUDA)
  install(
    TARGETS
      ${PROJECT_NAME}_library
      cuda_computer
      terrain_analyzer_post_processor_plugin
    EXPORT export_${PROJECT_NAME}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
  )
else()
  install(
    TARGETS
      ${PROJECT_NAME}_library
      terrain_analyzer_post_processor_plugin
    EXPORT export_${PROJECT_NAME}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
  )
endif()

install(
  DIRECTORY
    include/${PROJECT_NAME}/
  DESTINATION
    include/${PROJECT_NAME}
)

install(
  FILES post_processors.xml
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY
    params
    launch
  DESTINATION
    share/${PROJECT_NAME}
)

## Export
pluginlib_export_plugin_description_file(terrain_analyzer post_processors.xml)
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME}_library terrain_analyzer_post_processor_plugin)
ament_export_targets(export_${PROJECT_NAME})
ament_export_dependencies(
  rclcpp se2_grid_core se2_grid_ros se2_grid_msgs
  sensor_msgs std_msgs geometry_msgs visualization_msgs nav_msgs
  message_filters pcl_conversions cv_bridge pluginlib tf2_ros tf2_eigen
)

ament_package()
