#pragma once

#include <nav2_mppi_controller/critic_function.hpp>
#include <se2_grid_core/SE2Grid.hpp>
#include <se2_grid_msgs/msg/se2_grid.hpp>
#include <rclcpp/rclcpp.hpp>
#include <mutex>

namespace se2_critics
{

class SE2TraversabilityCritic : public mppi::critics::CriticFunction
{
public:
  void initialize() override;
  void score(mppi::CriticData & data) override;

private:
  void se2GridCallback(const se2_grid_msgs::msg::SE2Grid::SharedPtr msg);

  rclcpp::Subscription<se2_grid_msgs::msg::SE2Grid>::SharedPtr se2_grid_sub_;
  std::shared_ptr<se2_grid::SE2Grid> se2_grid_;
  std::mutex grid_mutex_;

  float collision_cost_{1e6f};
  int trajectory_point_step_{2};
  int power_{1};
  float weight_{1.0f};
};

}  // namespace se2_critics
