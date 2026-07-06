#pragma once

#include <string>

#include <Eigen/Dense>

#include <se2_grid_core/SE2Grid.hpp>
#include <se2_grid_core/iterators/iterators.hpp>

#include "terrain_analyzer/post_processors/PostProcessorBase.hpp"

namespace terrain_analyzer
{
    class NormalComputer : public terrain_analyzer::PostProcessor<se2_grid::SE2Grid>
    {
        public:

            NormalComputer() {}

            ~NormalComputer() override = default;

            bool configure() override
            {
                if (!PostProcessor::getParam(std::string("radius"), radius_))
                {
                    RCLCPP_ERROR(rclcpp::get_logger(\"PostProcessors\"), "NormalComputer did not find parameter `radius`.");
                    return false;
                }

                if (radius_ < 0.0)
                {
                    RCLCPP_ERROR(rclcpp::get_logger(\"PostProcessors\"), "NormalComputer: Radius must be greater than zero.");
                    return false;
                }
                RCLCPP_DEBUG(rclcpp::get_logger(\"PostProcessors\"), "Radius = %f.", radius_);

                if (!PostProcessor::getParam(std::string("input_layer"), inputLayer_))
                {
                    RCLCPP_ERROR(rclcpp::get_logger(\"PostProcessors\"), "NormalComputer did not find parameter `input_layer`.");
                    return false;
                }

                RCLCPP_DEBUG(rclcpp::get_logger(\"PostProcessors\"), "NormalComputer input layer is = %s.", inputLayer_.c_str());

                if (!PostProcessor::getParam(std::string("output_layers_prefix"), outputLayersPrefix_))
                {
                    RCLCPP_ERROR(rclcpp::get_logger(\"PostProcessors\"), "Step filter did not find parameter `output_layers_prefix`.");
                    return false;
                }

                if (!PostProcessor::getParam(std::string("output_curv_layer"), outputCurvLayer_))
                {
                    RCLCPP_ERROR(rclcpp::get_logger(\"PostProcessors\"), "Step filter did not find parameter `output_curv_layer`.");
                    return false;
                }

                RCLCPP_DEBUG(rclcpp::get_logger(\"PostProcessors\"), "NormalComputer output_layers_prefix = %s.", outputLayersPrefix_.c_str());

                return true;
            }

            bool process(const se2_grid::SE2Grid& mapIn, se2_grid::SE2Grid& mapOut) override
            {
                mapOut = mapIn;
                mapOut.add(std::string(outputLayersPrefix_+"x"), false);
                mapOut.add(std::string(outputLayersPrefix_+"y"), false);
                mapOut.add(std::string(outputLayersPrefix_+"z"), false);
                mapOut.add(std::string(outputCurvLayer_), false);

                for (se2_grid::SE2GridIterator iterator(mapOut); !iterator.isPastEnd(); ++iterator)
                {
                    if (mapOut.isValid(*iterator, inputLayer_))
                    {
                        const Eigen::Array3i index(*iterator);
                        Eigen::Vector3d center;
                        mapOut.index2Pos(index, center);

                        const double minRadius = 0.5 * mapOut.getResolutionPos();
                        if (radius_ <= minRadius)
                        {
                            RCLCPP_WARN(rclcpp::get_logger(\"PostProcessors\"), "Estimation radius is smaller than allowed by the mapOut resolution (%f < %f)", radius_, minRadius);
                        }

                        size_t nPoints = 0;
                        Eigen::Vector3d sum = Eigen::Vector3d::Zero();
                        Eigen::Matrix3d sumSquared = Eigen::Matrix3d::Zero();
                        double approx_curv = 0.0;
                        for (se2_grid::CircleIterator circleIterator(mapOut, center.head(2), radius_); !circleIterator.isPastEnd(); ++circleIterator)
                        {
                            Eigen::Vector3d point;
                            if (!mapOut.index2Pos(*circleIterator, point))
                                continue;
                            point[2] = mapOut.at(inputLayer_, *circleIterator);
                            nPoints++;
                            sum += point;
                            sumSquared.noalias() += point * point.transpose();
                        }

                        Eigen::Vector3d normalVector = Eigen::Vector3d::Zero();
                        if (nPoints < 3)
                        {
                            RCLCPP_DEBUG(rclcpp::get_logger(\"PostProcessors\"), "Not enough points to establish normal direction (nPoints = %i)", static_cast<int>(nPoints));
                            normalVector = Eigen::Vector3d::UnitZ();
                        } else
                        {
                            const Eigen::Vector3d mean = sum / nPoints;
                            const Eigen::Matrix3d covarianceMatrix = sumSquared / nPoints - mean * mean.transpose();

                            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver;
                            solver.computeDirect(covarianceMatrix, Eigen::DecompositionOptions::ComputeEigenvectors);
                            if (solver.eigenvalues()(1) > 1e-8)
                            {
                                normalVector = solver.eigenvectors().col(0);
                                approx_curv = solver.eigenvalues()(0) / solver.eigenvalues().sum() * 3.0;
                            }
                            else
                            {
                                RCLCPP_DEBUG(rclcpp::get_logger(\"PostProcessors\"), "Covariance matrix needed for eigen decomposition is degenerated.");
                                RCLCPP_DEBUG(rclcpp::get_logger(\"PostProcessors\"), "Expected cause: data is on a straight line (nPoints = %i)", static_cast<int>(nPoints));
                                normalVector = Eigen::Vector3d::UnitZ();
                            }
                        }

                        if (normalVector.dot(positive_axis) < 0.0)
                            normalVector = -normalVector;

                        mapOut.at(outputLayersPrefix_ + "x", index) = normalVector.x();
                        mapOut.at(outputLayersPrefix_ + "y", index) = normalVector.y();
                        mapOut.at(outputLayersPrefix_ + "z", index) = normalVector.z();
                        mapOut.at(outputCurvLayer_, index) = approx_curv;
                    }
                }

                return true;
            }

        private:

            double radius_ = 0.0;

            Eigen::Vector3d positive_axis = Eigen::Vector3d::UnitZ();

            std::string inputLayer_;

            std::string outputLayersPrefix_;

            std::string outputCurvLayer_;
    };

}  // namespace terrain_analyzer