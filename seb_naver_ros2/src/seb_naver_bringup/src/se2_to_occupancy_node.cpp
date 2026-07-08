#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <se2_grid_msgs/msg/se2_grid.hpp>
#include <se2_grid_core/SE2Grid.hpp>
#include <se2_grid_ros/se2_grid_ros.hpp>
#include <algorithm>
#include <cmath>

class SE2ToOccupancyNode : public rclcpp::Node
{
public:
  SE2ToOccupancyNode() : Node("se2_to_occupancy_node")
  {
    this->declare_parameter("input_topic", "/fused_map");
    this->declare_parameter("output_topic", "/map");
    this->declare_parameter("risk_threshold", 1.0);
    this->declare_parameter("layer_name", "risk");

    std::string input_topic = this->get_parameter("input_topic").as_string();
    std::string output_topic = this->get_parameter("output_topic").as_string();
    
    risk_threshold_ = this->get_parameter("risk_threshold").as_double();
    layer_name_ = this->get_parameter("layer_name").as_string();

    sub_ = this->create_subscription<se2_grid_msgs::msg::SE2Grid>(
      input_topic, 1, std::bind(&SE2ToOccupancyNode::gridCallback, this, std::placeholders::_1)
    );

    rclcpp::QoS map_qos(1);
    map_qos.transient_local();
    pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(output_topic, map_qos);
    
    RCLCPP_INFO(this->get_logger(), "Started SE2 to OccupancyGrid converter. %s -> %s", input_topic.c_str(), output_topic.c_str());
  }

private:
  void gridCallback(const se2_grid_msgs::msg::SE2Grid::SharedPtr msg)
  {
    se2_grid::SE2Grid se2_grid;
    if (!se2_grid::SE2GridRosConverter::fromMessage(*msg, se2_grid)) {
      RCLCPP_ERROR(this->get_logger(), "Failed to convert SE2Grid message");
      return;
    }

    if (!se2_grid.exists(layer_name_)) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, 
                           "Layer '%s' does not exist in SE2Grid", layer_name_.c_str());
      return;
    }

    nav_msgs::msg::OccupancyGrid occ_grid;
    occ_grid.header = msg->info.header;
    occ_grid.info.resolution = se2_grid.getResolutionPos();
    occ_grid.info.width = se2_grid.getSizePos().x();
    occ_grid.info.height = se2_grid.getSizePos().y();
    
    // Calculate the origin. SE2Grid position is the center. 
    // OccupancyGrid origin is the bottom-left corner of the grid.
    occ_grid.info.origin.position.x = se2_grid.getPosition().x() - (occ_grid.info.width * occ_grid.info.resolution) / 2.0;
    occ_grid.info.origin.position.y = se2_grid.getPosition().y() - (occ_grid.info.height * occ_grid.info.resolution) / 2.0;
    occ_grid.info.origin.position.z = 0.0;
    occ_grid.info.origin.orientation.w = 1.0;

    occ_grid.data.assign(occ_grid.info.width * occ_grid.info.height, -1);

    bool has_so2 = se2_grid.hasSO2(layer_name_);
    int size_yaw = se2_grid.getSizeYaw();
    
    // In se2_grid_core, grid indexing is (row, col) = (x, y) generally, but let's check carefully.
    // The underlying data is usually Eigen::MatrixXf which is accessed by (row, col).
    // In SE2Grid, size.x() corresponds to rows, size.y() corresponds to cols.
    // nav_msgs::OccupancyGrid data is row-major (y * width + x).
    
    for (unsigned int y = 0; y < occ_grid.info.height; ++y) {
      for (unsigned int x = 0; x < occ_grid.info.width; ++x) {
        Eigen::Array2i index(x, y); 
        
        float max_risk = -std::numeric_limits<float>::infinity();
        bool valid = false;

        if (has_so2) {
          for (int yaw = 0; yaw < size_yaw; ++yaw) {
            Eigen::Array3i idx3(x, y, yaw);
            if (se2_grid.isValid(idx3, {layer_name_})) {
              max_risk = std::max(max_risk, se2_grid.at(layer_name_, idx3));
              valid = true;
            }
          }
        } else {
          Eigen::Array3i idx3(x, y, 0);
          if (se2_grid.isValid(idx3, {layer_name_})) {
            max_risk = se2_grid.at(layer_name_, idx3);
            valid = true;
          }
        }

        int occ_idx = y * occ_grid.info.width + x;
        if (valid) {
          if (max_risk >= risk_threshold_) {
            occ_grid.data[occ_idx] = 100;
          } else {
            // Map [0, risk_threshold_) to [0, 99]
            int cost = static_cast<int>(std::max(0.0, (max_risk / risk_threshold_) * 99.0));
            occ_grid.data[occ_idx] = std::min(99, cost);
          }
        } else {
          occ_grid.data[occ_idx] = -1; // Unknown
        }
      }
    }

    pub_->publish(occ_grid);
  }

  rclcpp::Subscription<se2_grid_msgs::msg::SE2Grid>::SharedPtr sub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr pub_;
  double risk_threshold_;
  std::string layer_name_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SE2ToOccupancyNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
