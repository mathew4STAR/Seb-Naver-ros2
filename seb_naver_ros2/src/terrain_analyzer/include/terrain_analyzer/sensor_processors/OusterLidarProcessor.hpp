#pragma once

#include <terrain_analyzer/sensor_processors/SensorProcessorVirtual.hpp>

namespace terrain_analyzer 
{

    class OusterLidarProcessor : public SensorProcessorVirtual
    {
    public:

        OusterLidarProcessor() {}

        ~OusterLidarProcessor() noexcept override = default;

    private:

        bool initSensorParam(rclcpp::Node::SharedPtr node) override
        {
            beam_sigma2 = node->declare_parameter("sensor_processors.ouster.beam_sigma", 0.008);
            beam_sigma2 *= beam_sigma2;

            return true;
        }

        bool computeVariances(pcl::PointCloud<pcl::PointXYZ>::ConstPtr pointCloudSensor, 
                                const Eigen::Matrix4d& bodyTrans,
                                const Eigen::Matrix<double, 6, 6>& bodyPoseCovariance,
                                Eigen::VectorXf& variances) override
        {
            variances.resize(pointCloudSensor->size());
            Eigen::Vector3f bodyRTz = bodyTrans.topLeftCorner(3, 3).transpose().col(2).cast<float>();
            Eigen::Vector3f Js = sensor2BodyRf.transpose() * bodyRTz;
            Eigen::Vector3f Jr = Eigen::Vector3f::Zero();
            Eigen::Matrix3f SigmaS = Eigen::Matrix3f::Identity();
            Eigen::Matrix3f SigmaR = bodyPoseCovariance.bottomRightCorner(3, 3).cast<float>();
            float sigma2_bodyT = bodyPoseCovariance(2, 2);

            for (size_t i=0; i<pointCloudSensor->size(); i++)
            {
                const auto& point = pointCloudSensor->points[i];
                Eigen::Vector3f pointVec(point.x, point.y, point.z);
                Eigen::Vector3f pointVec2 = pointVec.cwiseAbs2();

                SigmaS = (pointVec2 / pointVec2.sum() * beam_sigma2).asDiagonal();
                Jr = skewSym(sensor2BodyRf * pointVec + sensor2BodyTf) * bodyRTz;
                variances(i) = sigma2_bodyT + Js.transpose() * SigmaS * Js
                             + Jr.transpose() * SigmaR * Jr;
            }

            return true;
        }

        double beam_sigma2;
    };

} /* namespace terrain_analyzer */
