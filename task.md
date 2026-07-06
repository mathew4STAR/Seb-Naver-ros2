# WP3 Sub-step 1: Port terrain_analyzer to ROS2

## Build System
- [x] Rewrite `CMakeLists.txt` (catkin → ament_cmake)
- [x] Rewrite `package.xml` (format 2/catkin → format 3/ament)

## Source Files
- [x] Modify `terrain_analyzer_node.cpp` (entry point)
- [x] Modify `TerrainAnalyzer.hpp` (main header)
- [x] Modify `TerrainAnalyzer.cpp` (main implementation — CUDA path only, #if 0 non-CUDA)
- [x] Modify `post_processors.cpp` (pluginlib macro)

## Launch & Params
- [x] Create `terrain_analyzer.launch.py` (ROS2 Python launch)
- [x] Reformat `elevation_map_gpu.yaml` for ROS2 param convention

## Remaining (next sub-step)
- [ ] Port `PostProcessorBase.hpp` (XmlRpc → yaml-cpp)
- [ ] Port `SensorProcessorVirtual.hpp` (ros::NodeHandle → rclcpp::Node)
- [ ] Port `OusterLidarProcessor.hpp` (same)
- [ ] Port all post_processor headers (ROS_ERROR → RCLCPP_ERROR)
- [ ] Build test with colcon
