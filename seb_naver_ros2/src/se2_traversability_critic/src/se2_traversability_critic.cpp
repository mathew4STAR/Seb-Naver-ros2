#include "se2_traversability_critic/se2_traversability_critic.hpp"
#include <se2_grid_ros/se2_grid_ros.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <cmath>

namespace se2_critics
{

void SE2TraversabilityCritic::initialize()
{
  auto node = parent_.lock();
  
  // Read parameters
  // Nav2 critic API usually expects parameters namespaced under the controller's namespace + critic name.
  // The 'name_' member comes from CriticFunction.
  auto param_ns = name_ + ".";
  
  nav2_util::declare_parameter_if_not_declared(
      node, param_ns + "collision_cost", rclcpp::ParameterValue(1e6f));
  nav2_util::declare_parameter_if_not_declared(
      node, param_ns + "trajectory_point_step", rclcpp::ParameterValue(2));
  nav2_util::declare_parameter_if_not_declared(
      node, param_ns + "power", rclcpp::ParameterValue(1));

  node->get_parameter(param_ns + "collision_cost", collision_cost_);
  node->get_parameter(param_ns + "trajectory_point_step", trajectory_point_step_);
  node->get_parameter(param_ns + "power", power_);

  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(weight_, "weight", 1.0f);

  // Using /fused_map as default based on the implementation plan
  se2_grid_sub_ = node->create_subscription<se2_grid_msgs::msg::SE2Grid>(
    "/fused_map", 1,
    std::bind(&SE2TraversabilityCritic::se2GridCallback, this, std::placeholders::_1)
  );

  RCLCPP_INFO(logger_, "SE2TraversabilityCritic initialized with topic: /fused_map");
}

void SE2TraversabilityCritic::se2GridCallback(const se2_grid_msgs::msg::SE2Grid::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(grid_mutex_);
  if (!se2_grid_) {
    se2_grid_ = std::make_shared<se2_grid::SE2Grid>();
  }
  se2_grid::SE2GridRosConverter::fromMessage(*msg, *se2_grid_);
}

void SE2TraversabilityCritic::score(mppi::CriticData & data)
{
  std::lock_guard<std::mutex> lock(grid_mutex_);
  if (!se2_grid_ || !se2_grid_->exists("risk")) {
    return; // Cannot evaluate without risk layer
  }

  auto & trajectories = data.trajectories; 
  auto & costs = data.costs;

  const auto batch_size = trajectories.x.shape(0);
  const auto time_steps = trajectories.x.shape(1);

  for (size_t i = 0; i < batch_size; ++i) {
    float traj_cost = 0.0f;
    bool in_collision = false;

    for (size_t t = 0; t < time_steps; t += trajectory_point_step_) {
      float x = trajectories.x(i, t);
      float y = trajectories.y(i, t);
      float yaw = trajectories.yaws(i, t);

      Eigen::Vector3d query(x, y, yaw);
      Eigen::Array3i index;

      if (se2_grid_->pos2Index(query, index)) {
        if (se2_grid_->isValid(index, std::string("risk"))) {
          float risk = se2_grid_->at("risk", index);
          if (risk >= 1.0f) {
            in_collision = true;
            break;
          }
          traj_cost += std::pow(risk, power_);
        }
      }
    }

    if (in_collision) {
      costs[i] += collision_cost_;
    } else {
      costs[i] += traj_cost * weight_;
    }
  }
}

}  // namespace se2_critics

// Register as a Nav2 MPPI critic plugin
PLUGINLIB_EXPORT_CLASS(se2_critics::SE2TraversabilityCritic, mppi::critics::CriticFunction)
