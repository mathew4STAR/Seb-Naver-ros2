#pragma once

#include <string>

#include <se2_grid_core/SE2Grid.hpp>
#include <se2_grid_core/iterators/iterators.hpp>

#include "terrain_analyzer/post_processors/PostProcessorBase.hpp"
#include "terrain_analyzer/utils/EigenLab.hpp"

namespace terrain_analyzer
{
    class MathProcessor : public terrain_analyzer::PostProcessor<se2_grid::SE2Grid>
    {
        public:

            MathProcessor() {}

            ~MathProcessor() override = default;

            bool configure() override
            {
                if (!PostProcessor::getParam(std::string("expression"), expression_))
                {
                    RCLCPP_ERROR(rclcpp::get_logger("PostProcessors"), "MathProcessor did not find parameter 'expression'.");
                    return false;
                }

                if (!PostProcessor::getParam(std::string("has_so2"), so2_))
                {
                    RCLCPP_ERROR(rclcpp::get_logger("PostProcessors"), "MathProcessor did not find parameter 'has_so2'.");
                    return false;
                }

                if (!PostProcessor::getParam(std::string("output_layer"), outputLayer_))
                {
                    RCLCPP_ERROR(rclcpp::get_logger("PostProcessors"), "MathProcessor did not find parameter 'output_layer'.");
                    return false;
                }

                return true;
            }

            bool process(const se2_grid::SE2Grid& mapIn, se2_grid::SE2Grid& mapOut) override
            {
                mapOut = mapIn;
                std::vector<Eigen::MatrixXf> add_data;
                for (int i=0; i<mapOut.getSizeYaw(); i++)
                {
                    for (const auto& layer : mapOut.getLayers())
                    {
                        if (mapOut.hasSO2(layer))
                            parser_.var(layer).setShared(mapOut[layer][i]);
                        else
                            parser_.var(layer).setShared(mapOut[layer][0]);
                    }
                    EigenLab::Value<Eigen::MatrixXf> result(parser_.eval(expression_));
                    add_data.push_back(result.matrix());
                    if (so2_)
                        break;
                }
                
                mapOut.add(outputLayer_, add_data);

                return true;
            }

        private:

            EigenLab::Parser<Eigen::MatrixXf> parser_;

            std::string expression_;

            std::string outputLayer_;

            bool so2_;
    };

}  // namespace terrain_analyzer