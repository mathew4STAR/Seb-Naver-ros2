#include <rclcpp/rclcpp.hpp>
#include <se2_grid_ros/SE2GridRosConverter.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <std_msgs/msg/float32.hpp>
#include <mutex>
#include <cmath>
#include <algorithm>

class SE2VisualizerNode : public rclcpp::Node
{
public:
  SE2VisualizerNode() : Node("se2_visualizer_node"), theta_(0.0)
  {
    se2_grid_sub_ = this->create_subscription<se2_grid_msgs::msg::SE2Grid>(
      "/fused_map", 1,
      std::bind(&SE2VisualizerNode::se2GridCallback, this, std::placeholders::_1)
    );

    theta_sub_ = this->create_subscription<std_msgs::msg::Float32>(
      "/se2_theta_slice", 1,
      std::bind(&SE2VisualizerNode::thetaCallback, this, std::placeholders::_1)
    );

    occ_grid_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/se2_slice_map", 1);
    
    // Publish at 2Hz for visual updates
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(500),
      std::bind(&SE2VisualizerNode::publishGrid, this)
    );

    RCLCPP_INFO(this->get_logger(), "SE2 Visualizer Node initialized. Waiting for /fused_map and /se2_theta_slice.");
  }

private:
  void se2GridCallback(const se2_grid_msgs::msg::SE2Grid::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(grid_mutex_);
    if (!se2_grid_) {
      se2_grid_ = std::make_shared<se2_grid::SE2Grid>();
    }
    se2_grid::SE2GridRosConverter::fromMessage(*msg, *se2_grid_);
  }

  void thetaCallback(const std_msgs::msg::Float32::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(grid_mutex_);
    theta_ = msg->data;
  }

  void publishGrid()
  {
    std::lock_guard<std::mutex> lock(grid_mutex_);
    if (!se2_grid_ || !se2_grid_->exists("risk")) {
      return;
    }

    nav_msgs::msg::OccupancyGrid grid_msg;
    grid_msg.header.stamp = this->now();
    grid_msg.header.frame_id = se2_grid_->getFrameId();

    grid_msg.info.resolution = se2_grid_->getResolutionPos();
    grid_msg.info.width = se2_grid_->getSizePos()(0);
    grid_msg.info.height = se2_grid_->getSizePos()(1);
    
    // OccupancyGrid origin is the bottom-left corner
    Eigen::Vector2d origin = se2_grid_->getPosition() - (se2_grid_->getLengthPos() / 2.0).matrix();
    grid_msg.info.origin.position.x = origin.x();
    grid_msg.info.origin.position.y = origin.y();
    grid_msg.info.origin.position.z = 0.0;
    grid_msg.info.origin.orientation.w = 1.0;

    grid_msg.data.resize(grid_msg.info.width * grid_msg.info.height, -1);
    
    // Iterate over cells and extract cost for the current theta_
    for (unsigned int y = 0; y < grid_msg.info.height; ++y) {
      for (unsigned int x = 0; x < grid_msg.info.width; ++x) {
        // Query by position coordinates to correctly use the SE2Grid indices
        Eigen::Vector3d pos;
        pos.x() = origin.x() + (x + 0.5) * grid_msg.info.resolution;
        pos.y() = origin.y() + (y + 0.5) * grid_msg.info.resolution;
        pos.z() = theta_;
        
        Eigen::Array3i idx;
        if (se2_grid_->pos2Index(pos, idx)) {
          if (se2_grid_->isValid(idx, "risk")) {
            float risk = se2_grid_->at("risk", idx);
            // risk >= 1.0 is considered lethal (100)
            int cost = std::min(100, std::max(0, static_cast<int>(risk * 100.0f)));
            grid_msg.data[y * grid_msg.info.width + x] = cost;
          }
        }
      }
    }

    occ_grid_pub_->publish(grid_msg);
  }

  rclcpp::Subscription<se2_grid_msgs::msg::SE2Grid>::SharedPtr se2_grid_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr theta_sub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr occ_grid_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::shared_ptr<se2_grid::SE2Grid> se2_grid_;
  std::mutex grid_mutex_;
  float theta_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SE2VisualizerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
