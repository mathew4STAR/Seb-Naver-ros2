#pragma once

#include <string>
#include <cmath>

#include <se2_grid_core/SE2Grid.hpp>
#include <se2_grid_core/iterators/iterators.hpp>

#include "terrain_analyzer/post_processors/PostProcessorBase.hpp"

/*
Z = Ax²y² + Bx²y + Cxy² + Dx² + Ey² + Fxy + Gx + Hy + I
A = [(Z1 + Z3 + Z7 + Z9) / 4  - (Z2 + Z4 + Z6 + Z8) / 2 + Z5] / L4
B = [(Z1 + Z3 - Z7 - Z9) /4 - (Z2 - Z8) /2] / L3
C = [(-Z1 + Z3 - Z7 + Z9) /4 + (Z4 - Z6)] /2] / L3
D = [(Z4 + Z6) /2 - Z5] / L2
E = [(Z2 + Z8) /2 - Z5] / L2
F = (-Z1 + Z3 + Z7 - Z9) / 4L2
G = (-Z4 + Z6) / 2L
H = (Z2 - Z8) / 2L
I = Z5
Curvature = -2(D + E) * 100
*/

namespace terrain_analyzer
{
    class CurvatureComputer : public terrain_analyzer::PostProcessor<se2_grid::SE2Grid>
    {
        public:

            CurvatureComputer() {}

            ~CurvatureComputer() override = default;

            bool configure() override
            {
                if (!PostProcessor::getParam(std::string("input_layer"), inputLayer_))
                {
                    RCLCPP_ERROR(rclcpp::get_logger("PostProcessors"), "CurvatureComputer did not find parameter `input_layer`.");
                    return false;
                }

                RCLCPP_DEBUG(rclcpp::get_logger("PostProcessors"), "CurvatureComputer input layer is = %s.", inputLayer_.c_str());

                if (!PostProcessor::getParam(std::string("output_layer"), outputLayer_))
                {
                    RCLCPP_ERROR(rclcpp::get_logger("PostProcessors"), "CurvatureComputer did not find parameter `output_layer`.");
                    return false;
                }

                if (!PostProcessor::getParam(std::string("max_cur"), max_cur_))
                {
                    RCLCPP_ERROR(rclcpp::get_logger("PostProcessors"), "CurvatureComputer did not find parameter `max_cur`.");
                    return false;
                }

                if (!PostProcessor::getParam(std::string("min_cur"), min_cur_))
                {
                    RCLCPP_ERROR(rclcpp::get_logger("PostProcessors"), "CurvatureComputer did not find parameter `min_cur`.");
                    return false;
                }

                RCLCPP_DEBUG(rclcpp::get_logger("PostProcessors"), "CurvatureComputer output_layer = %s.", outputLayer_.c_str());

                return true;
            }

            bool process(const se2_grid::SE2Grid& mapIn, se2_grid::SE2Grid& mapOut) override
            {
                if (!mapIn.isDefaultStartIndex())
                    throw std::runtime_error("CurvatureComputer cannot be used with maps that don't have a default buffer start index.");

                mapOut = mapIn;
                mapOut.add(outputLayer_, false);
                auto& input = mapOut[inputLayer_][0];
                auto& curvature = mapOut[outputLayer_][0];
                const float L2 = mapOut.getResolutionPos() * mapOut.getResolutionPos();

                for (Eigen::Index j{0}; j < input.cols(); ++j)
                    for (Eigen::Index i{0}; i < input.rows(); ++i)
                    {
                        if (!std::isfinite(input(i, j)))
                            continue;
                        
                        float D = ((input(i, j == 0 ? j : j - 1) + input(i, j == input.cols() - 1 ? j : j + 1)) / 2.0 - input(i, j)) / L2;
                        float E = ((input(i == 0 ? i : i - 1, j) + input(i == input.rows() - 1 ? i : i + 1, j)) / 2.0 - input(i, j)) / L2;
                        if (!std::isfinite(D))
                            D = 0.0;
                        if (!std::isfinite(E))
                            E = 0.0;
                        curvature(i, j) = std::min(std::max(-2.0 * (D + E), min_cur_), max_cur_);
                    }

                return true;
            }

        private:

            double max_cur_ = 0.0;

            double min_cur_ = 0.0;

            std::string inputLayer_;

            std::string outputLayer_;
    };

}  // namespace terrain_analyzer