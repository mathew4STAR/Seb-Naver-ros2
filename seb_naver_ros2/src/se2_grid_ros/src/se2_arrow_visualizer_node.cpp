#include <rclcpp/rclcpp.hpp>
#include <se2_grid_ros/SE2GridRosConverter.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <mutex>
#include <cmath>
#include <algorithm>

class SE2ArrowVisualizerNode : public rclcpp::Node
{
public:
  SE2ArrowVisualizerNode() : Node("se2_arrow_visualizer_node")
  {
    se2_grid_sub_ = this->create_subscription<se2_grid_msgs::msg::SE2Grid>(
      "/fused_map", 1,
      std::bind(&SE2ArrowVisualizerNode::se2GridCallback, this, std::placeholders::_1)
    );

    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/se2_arrows", 1);
    
    // Publish at 2Hz for visual updates
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(500),
      std::bind(&SE2ArrowVisualizerNode::publishArrows, this)
    );

    RCLCPP_INFO(this->get_logger(), "SE2 Arrow Visualizer Node initialized.");
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

  void publishArrows()
  {
    std::lock_guard<std::mutex> lock(grid_mutex_);
    if (!se2_grid_ || !se2_grid_->exists("risk")) {
      return;
    }

    visualization_msgs::msg::MarkerArray marker_array;
    auto stamp = this->now();
    std::string frame_id = se2_grid_->getFrameId();

    Eigen::Vector2d origin = se2_grid_->getPosition() - (se2_grid_->getLengthPos() / 2.0).matrix();
    double res_pos = se2_grid_->getResolutionPos();
    int width = se2_grid_->getSizePos()(0);
    int height = se2_grid_->getSizePos()(1);

    // Delete previous markers
    visualization_msgs::msg::Marker delete_marker;
    delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_array.markers.push_back(delete_marker);

    int marker_id = 0;
    
    // 1. Angle rounding increment (increase to 15 degrees to reduce arrows)
    double angle_increment = 15.0 * M_PI / 180.0;
    
    // 2. Space downsampling (skip every N cells)
    int space_step = 2; 

    // 3. Cost threshold (only show arrows for significant costs)
    int min_cost_threshold = 30; 
    
    double scale_k = 0.005; // Scaling factor for the arrow length

    for (int y = 0; y < height; y += space_step) {
      for (int x = 0; x < width; x += space_step) {
        
        // Base coordinate of the grid cell
        double cx = origin.x() + (x + 0.5) * res_pos;
        double cy = origin.y() + (y + 0.5) * res_pos;
        
        // Iterate over rounded angles
        for (double theta = 0; theta < 2.0 * M_PI; theta += angle_increment) {
          
          Eigen::Vector3d pos(cx, cy, theta);
          Eigen::Array3i idx;
          
          if (se2_grid_->pos2Index(pos, idx)) {
            if (se2_grid_->isValid(idx, "risk")) {
              float risk = se2_grid_->at("risk", idx);
              
              // Scale risk to cost
              int cost = std::max(0, static_cast<int>(std::floor(risk * 100.0f)));
              
              // Only draw if cost is above threshold
              if (cost > min_cost_threshold) {
                visualization_msgs::msg::Marker marker;
                marker.header.frame_id = frame_id;
                marker.header.stamp = stamp;
                marker.ns = "se2_arrows";
                marker.id = marker_id++;
                marker.type = visualization_msgs::msg::Marker::ARROW;
                marker.action = visualization_msgs::msg::Marker::ADD;
                
                // Arrow origin
                geometry_msgs::msg::Point p1, p2;
                p1.x = cx;
                p1.y = cy;
                p1.z = 0.0;
                
                // Arrow tip
                double length = cost * scale_k;
                p2.x = cx + length * std::cos(theta);
                p2.y = cy + length * std::sin(theta);
                p2.z = 0.0;
                
                marker.points.push_back(p1);
                marker.points.push_back(p2);
                
                // Shaft diameter, head diameter, head length
                marker.scale.x = 0.02; // shaft diameter
                marker.scale.y = 0.04; // head diameter
                marker.scale.z = 0.04; // head length
                
                // Color (e.g. red, with some transparency based on cost, or just solid)
                marker.color.a = 0.8;
                marker.color.r = 1.0;
                marker.color.g = 0.0;
                marker.color.b = 0.0;
                
                marker_array.markers.push_back(marker);
              }
            }
          }
        }
      }
    }

    if (!marker_array.markers.empty()) {
      marker_pub_->publish(marker_array);
    }
  }

  rclcpp::Subscription<se2_grid_msgs::msg::SE2Grid>::SharedPtr se2_grid_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::shared_ptr<se2_grid::SE2Grid> se2_grid_;
  std::mutex grid_mutex_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SE2ArrowVisualizerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
