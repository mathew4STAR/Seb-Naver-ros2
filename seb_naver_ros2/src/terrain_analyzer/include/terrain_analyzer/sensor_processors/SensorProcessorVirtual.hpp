#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <memory>
#include <string>

namespace terrain_analyzer
{

class SensorProcessorVirtual
{
    public:
        using Ptr = std::unique_ptr<SensorProcessorVirtual>;

        SensorProcessorVirtual() {}

        virtual ~SensorProcessorVirtual() = default;

        bool process(pcl::PointCloud<pcl::PointXYZ> cloud_world,
                    const Eigen::Matrix4d& bodyTrans,
                    const Eigen::Matrix<double, 6, 6>& bodyPoseCovariance,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloudWorld, 
                    Eigen::VectorXf& variances)
        {
            pcl::PointCloud<pcl::PointXYZ> temp_cloud;
            pcl::PointCloud<pcl::PointXYZ> cloud_sensor;

            // Remove NaN
            if (!cloud_world.is_dense)
            {
                std::vector<int> indices;
                pcl::removeNaNFromPointCloud(cloud_world, temp_cloud, indices);
                temp_cloud.is_dense = true;
                cloud_world.swap(temp_cloud);
            }

            // Voxel Grid Filter
            if (use_voxel_filter)
            {
                pcl::VoxelGrid<pcl::PointXYZ> voxelGridFilter;
                voxelGridFilter.setInputCloud(cloud_world.makeShared());
                voxelGridFilter.setLeafSize(voxel_size, voxel_size, voxel_size);
                voxelGridFilter.filter(temp_cloud);
                cloud_world.swap(temp_cloud);
            }

            // Transform to Body and Filter Z-axis
            pointCloudWorld->clear();
            Eigen::Affine3f bodyTransAf(bodyTrans.cast<float>());
            pcl::transformPointCloud(cloud_world, temp_cloud, bodyTransAf.inverse());
            for (int i=0; i<temp_cloud.size(); i++)
            {
                if (temp_cloud.points[i].z > ignore_z_min && temp_cloud.points[i].z < ignore_z_max)
                    pointCloudWorld->points.emplace_back(temp_cloud.points[i]);
            }
            pointCloudWorld->is_dense = true;
            pointCloudWorld->width = pointCloudWorld->points.size();
            pointCloudWorld->height = 1;
            pointCloudWorld->header.frame_id = "world";
            
            // Transform to Sensor and Map
            pcl::transformPointCloud(*pointCloudWorld, cloud_sensor, body2SensorTrans);
            pcl::transformPointCloud(*pointCloudWorld, *pointCloudWorld, bodyTransAf);

            return computeVariances(cloud_sensor.makeShared(), bodyTrans, bodyPoseCovariance, variances);
        }

        bool init(rclcpp::Node::SharedPtr node)
        {
            std::vector<double> sensor2BodyR;
            std::vector<double> sensor2BodyT;
            Eigen::Matrix4f body2SensorRT = Eigen::Matrix4f::Identity();

            use_voxel_filter = node->declare_parameter("sensor_processors.use_voxel_filter", false);
            voxel_size = node->declare_parameter("sensor_processors.voxel_size", 0.05);
            ignore_z_max = node->declare_parameter("sensor_processors.ignore_z_max", 0.5);
            ignore_z_min = node->declare_parameter("sensor_processors.ignore_z_min", -2.0);
            sensor2BodyR = node->declare_parameter("sensor_processors.sensor2BodyR", std::vector<double>{1,0,0, 0,1,0, 0,0,1});
            sensor2BodyT = node->declare_parameter("sensor_processors.sensor2BodyT", std::vector<double>{0,0,0});
            for (int i=0; i<3; i++)
            {
                for (int j=0; j<3; j++)
                    body2SensorRT(j, i) = sensor2BodyR[i*3+j];
                body2SensorRT(3, i) = -sensor2BodyT[i];
            }
            sensor2BodyRf = body2SensorRT.topLeftCorner(3, 3).transpose();
            sensor2BodyTf = body2SensorRT.topRightCorner(3, 1);
            body2SensorRT.topRightCorner(3, 1) = body2SensorRT.topLeftCorner(3, 3) * body2SensorRT.topRightCorner(3, 1);
            body2SensorTrans = Eigen::Affine3f(body2SensorRT);

            return initSensorParam(node);
        }

        Eigen::Matrix3f skewSym(Eigen::Vector3f vec)
        {
            Eigen::Matrix3f skem_sym;
            skem_sym << 0.0    , -vec(2), vec(1) , \
                        vec(2) , 0.0    , -vec(0), \
                        -vec(1), vec(0) , 0.0       ;
            return skem_sym;
        }

        virtual bool initSensorParam(rclcpp::Node::SharedPtr node) { return true; }

        virtual bool computeVariances(pcl::PointCloud<pcl::PointXYZ>::ConstPtr pointCloudSensor, 
                                        const Eigen::Matrix4d& bodyTrans,
                                        const Eigen::Matrix<double, 6, 6>& bodyPoseCovariance,
                                        Eigen::VectorXf& variances) = 0;

    protected:
        Eigen::Affine3f body2SensorTrans;

        Eigen::Matrix3f sensor2BodyRf;

        Eigen::Vector3f sensor2BodyTf;

        bool use_voxel_filter;

        double voxel_size;

        double ignore_z_max;

        double ignore_z_min;
};

} /* namespace terrain_analyzer */
