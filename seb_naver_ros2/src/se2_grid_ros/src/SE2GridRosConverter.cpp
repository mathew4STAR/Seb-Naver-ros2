#include "se2_grid_ros/SE2GridRosConverter.hpp"
#include "se2_grid_ros/SE2GridMsgHelpers.hpp"

// ROS2
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/quaternion.hpp>

// STL
#include <limits>
#include <algorithm>
#include <vector>

using namespace std;
using namespace Eigen;

namespace se2_grid {

SE2GridRosConverter::SE2GridRosConverter()
{
}

SE2GridRosConverter::~SE2GridRosConverter()
{
}

bool SE2GridRosConverter::fromMessage(const se2_grid_msgs::msg::SE2Grid& message, se2_grid::SE2Grid& SE2Grid, const std::vector<std::string>& /*layers*/)
{
  SE2Grid.initGeometry(Eigen::Array2d(message.info.length_x, message.info.length_y), 
                       message.info.pos_resolution, message.info.yaw_resolution,
                       Eigen::Vector2d(message.info.pose.position.x, message.info.pose.position.y));

  // Copy non-basic layers.
  unsigned int data_idx = 0;
  for (unsigned int i = 0u; i < message.layers.size(); ++i)
  {
    // TODO Could we use the data mapping (instead of copying) method here?
    std::vector<Eigen::MatrixXf> datas;
    int size_yaw = SE2Grid.getSizeYaw();
    if (message.has_so2[i])
    {
      for (int j=0; j<size_yaw; j++)
      {
        Eigen::MatrixXf d;
        if(!multiArrayMessageCopyToMatrixEigen(message.data[data_idx++], d))
        {
          return false;
        }
        datas.emplace_back(d);
      }
      SE2Grid.add(message.layers[i], datas);
    }
    else
    {
      Eigen::MatrixXf d;
      if(!multiArrayMessageCopyToMatrixEigen(message.data[data_idx++], d))
      {
        return false;
      }
      datas.emplace_back(d);
      SE2Grid.add(message.layers[i], datas);
    }
  }

  SE2Grid.setStartIndex(Eigen::Array2i(message.start_index_row, message.start_index_col));

  return true;
}

bool SE2GridRosConverter::fromMessage(const se2_grid_msgs::msg::SE2Grid& message, se2_grid::SE2Grid& SE2Grid)
{
  return fromMessage(message, SE2Grid, std::vector<std::string>());
}

void SE2GridRosConverter::toMessage(const se2_grid::SE2Grid& SE2Grid, se2_grid_msgs::msg::SE2Grid& message)
{
  toMessage(SE2Grid, SE2Grid.getLayers(), message);
}

void SE2GridRosConverter::toMessage(const se2_grid::SE2Grid& SE2Grid, const std::vector<std::string>& layers,
                                    se2_grid_msgs::msg::SE2Grid& message)
{
  rclcpp::Clock clock(RCL_ROS_TIME);
  message.info.header.stamp = clock.now();
  message.info.header.frame_id = SE2Grid.getFrameId();
  message.info.pos_resolution = SE2Grid.getResolutionPos();
  message.info.yaw_resolution = SE2Grid.getResolutionYaw();
  message.info.length_x = SE2Grid.getLengthPos().x();
  message.info.length_y = SE2Grid.getLengthPos().y();
  message.info.length_yaw = SE2Grid.getLengthYaw();
  message.info.pose.position.x = SE2Grid.getPosition().x();
  message.info.pose.position.y = SE2Grid.getPosition().y();
  message.info.pose.position.z = 0.0;
  message.info.pose.orientation.x = 0.0;
  message.info.pose.orientation.y = 0.0;
  message.info.pose.orientation.z = 0.0;
  message.info.pose.orientation.w = 1.0;

  message.layers = layers;

  message.data.clear();
  message.has_so2.clear();
  for (size_t i=0; i<layers.size(); i++)
  {
    std_msgs::msg::Float32MultiArray dataArray;
    if (SE2Grid.hasSO2(layers[i]))
      message.has_so2.emplace_back(true);
    else
      message.has_so2.emplace_back(false);
    
    std::vector<Eigen::MatrixXf> l = SE2Grid[layers[i]];
    for (size_t j=0; j<l.size(); j++)
    {
      matrixEigenCopyToMultiArrayMessage(l[j], dataArray);
      message.data.push_back(dataArray);
    }
  }

  message.start_index_row = SE2Grid.getStartIndex()(0);
  message.start_index_col = SE2Grid.getStartIndex()(1);
}

// NOTE: saveToBag() and loadFromBag() removed.
// In ROS2, use rosbag2_cpp or `ros2 bag` CLI if bag recording is needed.

} /* namespace */
