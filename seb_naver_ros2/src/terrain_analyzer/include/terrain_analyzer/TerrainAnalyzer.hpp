#pragma once

#include <se2_grid_core/SE2Grid.hpp>
#include <se2_grid_ros/se2_grid_ros.hpp>

// ROS2
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <message_filters/cache.h>
#include <message_filters/subscriber.h>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.hpp>

// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

// STL (replaced boost::thread with std::thread)
#include <thread>

#include "terrain_analyzer/sensor_processors/sensor_processors.hpp"
#include "terrain_analyzer/post_processors/post_processors.hpp"
#include "terrain_analyzer/utils/utils.hpp"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>

namespace terrain_analyzer
{
    struct ClampVar
    {
        ClampVar(const float& minVar, const float& maxVar) : minVar_(minVar), maxVar_(maxVar) {}
        const float operator()(const float& x) const
        {
            return x < minVar_ ? minVar_ : (x > maxVar_ ? std::numeric_limits<float>::infinity() : x);
        }
        float minVar_, maxVar_;
    };

    class TerrainAnalyzer
    {
        public:
#define ENABLE_CUDA 1
#ifdef ENABLE_CUDA
            TerrainAnalyzer() : 
                fused_map({"elevation", "var", "inpainted", "smooth", "normal_x", "normal_y", "risk", "zbx", "zby"}, 
                          {false, false, false, false, false, false, true, true, true}),
                sdf_map({"sdf"}, {false}),
                post_processors("se2_grid::SE2Grid") {}
#else
            TerrainAnalyzer() : 
                raw_map({"elevation", "var", "var_x", "var_y", "var_xy"}, 
                        {false, false, false, false, false}),
                fused_map({"elevation", "upper_bound", "lower_bound"},
                        {false, false, false}),
                post_processors("se2_grid::SE2Grid")
            {
                raw_basic_layers.clear();
                fused_basic_layers.clear();
                raw_basic_layers.emplace_back(std::string("elevation"));
                raw_basic_layers.emplace_back(std::string("var"));
                fused_basic_layers.emplace_back(std::string("elevation"));
                fused_basic_layers.emplace_back(std::string("lower_bound"));
                fused_basic_layers.emplace_back(std::string("upper_bound"));
            }
#endif

            // ROS2: init takes a shared_ptr to the Node
            bool init(rclcpp::Node::SharedPtr node);
            bool add(const pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloudMap, Eigen::VectorXf& variances);
            bool fuseAll();

            // ROS2: timer callback has no TimerEvent argument
            void mapCallback();
            // ROS2: SharedPtr instead of ConstPtr
            void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg); 

            inline float cumulativeDistributionFunction(float x, float mean, float standardDeviation)
            {
                return 0.5 * erfc(-(x - mean) / (standardDeviation * sqrt(2.0)));
            }
            
        private:
            // Store the node handle for parameter access, logging, clock, etc.
            rclcpp::Node::SharedPtr node_;

            // params
            int cache_size = 200;
            bool verbose = false;
            double min_var;
            double max_var;
            double min_var_hori;
            double max_var_hori;
            double mahalanobis_threshold;
            sensor_msgs::msg::PointCloud2 global_cloud_msg;

            // data
            message_filters::Cache<nav_msgs::msg::Odometry> odom_cache;
            OusterLidarProcessor sensor_processor;
            PostProcessorChain<se2_grid::SE2Grid> post_processors;
            se2_grid::SE2Grid raw_map;
            se2_grid::SE2Grid fused_map;
            se2_grid::SE2Grid sdf_map;
            std::vector<std::string> raw_basic_layers;
            std::vector<std::string> fused_basic_layers;

            // ROS2 pub/sub/timer
            message_filters::Subscriber<nav_msgs::msg::Odometry> odom_sub;
            rclcpp::Publisher<se2_grid_msgs::msg::SE2Grid>::SharedPtr       raw_pub;
            rclcpp::Publisher<se2_grid_msgs::msg::SE2Grid>::SharedPtr       fused_pub;
            rclcpp::Publisher<se2_grid_msgs::msg::SE2Grid>::SharedPtr       sdf_pub;
            rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr   normal_pub;
            rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr     shit_pub;
            rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr  cloud_sub;
            std::shared_ptr<tf2_ros::Buffer>                                tf_buffer_;
            std::shared_ptr<tf2_ros::TransformListener>                     tf_listener_;
            rclcpp::TimerBase::SharedPtr                                    map_timer;
            rclcpp::Time                                                    last_update_time;
    };
}