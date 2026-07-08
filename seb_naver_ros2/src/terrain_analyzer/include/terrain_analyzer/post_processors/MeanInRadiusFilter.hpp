#pragma once

#include <string>

#include <se2_grid_core/SE2Grid.hpp>
#include <se2_grid_core/iterators/iterators.hpp>

#include "terrain_analyzer/post_processors/PostProcessorBase.hpp"

namespace terrain_analyzer
{
    class MeanInRadiusFilter : public terrain_analyzer::PostProcessor<se2_grid::SE2Grid>
    {
        public:

            MeanInRadiusFilter() {}

            ~MeanInRadiusFilter() override = default;

            bool configure() override
            {
                if (!PostProcessor::getParam(std::string("radius"), radius_))
                {
                    RCLCPP_ERROR(rclcpp::get_logger("PostProcessors"), "MinInRadius filter did not find parameter `radius`.");
                    return false;
                }

                if (radius_ < 0.0)
                {
                    RCLCPP_ERROR(rclcpp::get_logger("PostProcessors"), "MinInRadius filter: Radius must be greater than zero.");
                    return false;
                }
                RCLCPP_DEBUG(rclcpp::get_logger("PostProcessors"), "Radius = %f.", radius_);

                if (!PostProcessor::getParam(std::string("input_layer"), inputLayer_))
                {
                    RCLCPP_ERROR(rclcpp::get_logger("PostProcessors"), "MinInRadius filter did not find parameter `input_layer`.");
                    return false;
                }

                RCLCPP_DEBUG(rclcpp::get_logger("PostProcessors"), "MinInRadius input layer is = %s.", inputLayer_.c_str());

                if (!PostProcessor::getParam(std::string("output_layer"), outputLayer_))
                {
                    RCLCPP_ERROR(rclcpp::get_logger("PostProcessors"), "Step filter did not find parameter `output_layer`.");
                    return false;
                }

                RCLCPP_DEBUG(rclcpp::get_logger("PostProcessors"), "MinInRadius output_layer = %s.", outputLayer_.c_str());

                return true;
            }

            bool process(const se2_grid::SE2Grid& mapIn, se2_grid::SE2Grid& mapOut) override
            {
                mapOut = mapIn;
                if (mapOut.hasSO2(inputLayer_))
                    mapOut.add(outputLayer_, true);
                else
                    mapOut.add(outputLayer_, false);

                double value{NAN};

                for (se2_grid::SE2GridIterator iterator(mapOut); !iterator.isPastEnd(); ++iterator)
                {
                    for (int i=0; i<mapOut[inputLayer_].size(); i++)
                    {
                        double valueSum = 0.0;
                        int counter = 0;

                        Eigen::Vector3d center;
                        mapOut.index2Pos(*iterator, center);

                        for (se2_grid::CircleIterator submapIterator(mapOut, center.head(2), radius_); !submapIterator.isPastEnd(); ++submapIterator)
                        {
                            Eigen::Array3i cidx = *submapIterator;
                            cidx(2) = i;

                            if (!mapOut.isValid(cidx, inputLayer_))
                                continue;

                            value = mapOut.at(inputLayer_, cidx);
                            valueSum += value;
                            counter++;
                        }

                        Eigen::Array3i idx = *iterator;
                        idx(2) = i;
                        if (counter != 0)
                            mapOut.at(outputLayer_, idx) = valueSum / (1.0 * counter);
                    }
                }

                return true;
            }

        private:

            double radius_ = 0.0;

            std::string inputLayer_;

            std::string outputLayer_;
    };

}  // namespace terrain_analyzer