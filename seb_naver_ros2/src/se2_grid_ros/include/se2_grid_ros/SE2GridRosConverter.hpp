#pragma once

#include <se2_grid_core/SE2Grid.hpp>
#include <se2_grid_msgs/msg/se2_grid.hpp>

// STL
#include <vector>
#include <unordered_map>

// Eigen
#include <Eigen/Core>

// ROS2
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/clock.hpp>

namespace se2_grid {

/*!
 * ROS2 interface for the Grid Map library.
 */
class SE2GridRosConverter
{
 public:
  /*!
   * Default constructor.
   */
  SE2GridRosConverter();

  /*!
   * Destructor.
   */
  virtual ~SE2GridRosConverter();

  /*!
   * Converts a ROS2 grid map message to a grid map object.
   * @param[in] message the grid map message.
   * @param[in] layers the layers to be copied.
   * @param[out] SE2Grid the grid map object to be initialized.
   * @return true if successful, false otherwise.
   */
  static bool fromMessage(const se2_grid_msgs::msg::SE2Grid& message, se2_grid::SE2Grid& SE2Grid, const std::vector<std::string>& layers);

  /*!
   * Converts a ROS2 grid map message to a grid map object.
   * @param[in] message the grid map message.
   * @param[out] SE2Grid the grid map object to be initialized.
   * @return true if successful, false otherwise.
   */
  static bool fromMessage(const se2_grid_msgs::msg::SE2Grid& message, se2_grid::SE2Grid& SE2Grid);

  /*!
   * Converts all layers of a grid map object to a ROS2 grid map message.
   * @param[in] SE2Grid the grid map object.
   * @param[out] message the grid map message to be populated.
   */
  static void toMessage(const se2_grid::SE2Grid& SE2Grid, se2_grid_msgs::msg::SE2Grid& message);

  /*!
   * Converts requested layers of a grid map object to a ROS2 grid map message.
   * @param[in] SE2Grid the grid map object.
   * @param[in] layers the layers to be added to the message.
   * @param[out] message the grid map message to be populated.
   */
  static void toMessage(const se2_grid::SE2Grid& SE2Grid, const std::vector<std::string>& layers,
                        se2_grid_msgs::msg::SE2Grid& message);

  // NOTE: saveToBag / loadFromBag removed.
  // In ROS2, use `ros2 bag` CLI or rosbag2_cpp directly if needed.

};

} /* namespace */
