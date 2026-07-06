#pragma once

#include <string>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include <se2_grid_core/SE2Grid.hpp>
#include <se2_grid_core/iterators/iterators.hpp>
#include <se2_grid_core/eigen_plugins/Functors.hpp>

#include "terrain_analyzer/post_processors/PostProcessorBase.hpp"

namespace terrain_analyzer
{
    class InpaintProcessor : public terrain_analyzer::PostProcessor<se2_grid::SE2Grid>
    {
        public:

            InpaintProcessor() {}

            ~InpaintProcessor() override = default;

            bool configure() override
            {
                if (!PostProcessor::getParam(std::string("radius"), radius_))
                {
                    RCLCPP_ERROR(rclcpp::get_logger(\"PostProcessors\"), "Inpaint processor did not find parameter `radius`.");
                    return false;
                }

                if (radius_ < 0.0)
                {
                    RCLCPP_ERROR(rclcpp::get_logger(\"PostProcessors\"), "Inpaint processor: Radius must be greater than zero.");
                    return false;
                }
                RCLCPP_DEBUG(rclcpp::get_logger(\"PostProcessors\"), "Radius = %f.", radius_);

                if (!PostProcessor::getParam(std::string("input_layer"), inputLayer_))
                {
                    RCLCPP_ERROR(rclcpp::get_logger(\"PostProcessors\"), "Inpaint processor did not find parameter `input_layer`.");
                    return false;
                }

                RCLCPP_DEBUG(rclcpp::get_logger(\"PostProcessors\"), "MinInRadius input layer is = %s.", inputLayer_.c_str());

                if (!PostProcessor::getParam(std::string("output_layer"), outputLayer_))
                {
                    RCLCPP_ERROR(rclcpp::get_logger(\"PostProcessors\"), "Step filter did not find parameter `output_layer`.");
                    return false;
                }

                RCLCPP_DEBUG(rclcpp::get_logger(\"PostProcessors\"), "MinInRadius output_layer = %s.", outputLayer_.c_str());

                return true;
            }

            bool process(const se2_grid::SE2Grid& mapIn, se2_grid::SE2Grid& mapOut) override
            {
                mapOut = mapIn;
                if (mapOut.hasSO2(inputLayer_))
                {
                    mapOut.add(outputLayer_, true);
                    mapOut.add("inpaint_mask", true, 0.0);
                }
                else
                {
                    mapOut.add(outputLayer_, false);
                    mapOut.add("inpaint_mask", false, 0.0);
                }

                std::vector<Eigen::MatrixXf> new_datas;
                for (int i=0; i<mapOut[inputLayer_].size(); i++)
                {
                    for (se2_grid::SE2GridIterator iterator(mapOut); !iterator.isPastEnd(); ++iterator)
                    {
                        Eigen::Array3i idx = *iterator;
                        idx(2) = i;
                        if (!mapOut.isValid(idx, inputLayer_))
                            mapOut.at("inpaint_mask", idx) = 1.0;
                    }

                    cv::Mat originalImage;
                    cv::Mat mask;
                    cv::Mat filledImage;
                    const float minValue = mapOut[inputLayer_][i].minCoeffOfFinites();
                    const float maxValue = mapOut[inputLayer_][i].maxCoeffOfFinites();

                    toImage(mapOut, inputLayer_, i, CV_8UC1, minValue, maxValue, originalImage);
                    toImage(mapOut, "inpaint_mask", i, CV_8UC1, mask);

                    const double radiusInPixels = radius_ / mapIn.getResolutionPos();
                    cv::inpaint(originalImage, mask, filledImage, radiusInPixels, cv::INPAINT_NS);

                    Eigen::MatrixXf data = mapOut[inputLayer_][i];
                    const float valueDiffer = maxValue - minValue;
                    float maxImageValue = (float)std::numeric_limits<unsigned char>::max();

                    for (se2_grid::SE2GridIterator iterator(mapOut); !iterator.isPastEnd(); ++iterator)
                    {
                        Eigen::Array3i idx = *iterator;
                        Eigen::Array3i imageIndex = iterator.getUnwrappedIndex();
                        const unsigned char imageValue = filledImage.at<unsigned char>(imageIndex(0), imageIndex(1));
                        const float mapValue = minValue + valueDiffer * ((float) imageValue / maxImageValue);
                        data(idx(0), idx(1)) = mapValue;
                    }

                    new_datas.emplace_back(data);
                }

                mapOut.add(outputLayer_, new_datas);
                mapOut.erase("inpaint_mask");

                return true;
            }
            
            bool toImage(const se2_grid::SE2Grid& se2Grid, const std::string& layer, const size_t idx,
                        const int encoding, const float lowerValue, const float upperValue,
                        cv::Mat& image)
            {
                Eigen::Array2i map_size = se2Grid.getSizePos();
                if (map_size(0) > 0 && map_size(1) > 0)
                {
                    image = cv::Mat::zeros(map_size(0), map_size(1), encoding);
                } else 
                {
                    std::cerr << "Invalid se2 grid?" << std::endl;
                    return false;
                }

                unsigned char imageMax = std::numeric_limits<unsigned char>::max();
                se2_grid::SE2Grid map = se2Grid;

                map[layer][idx] = map[layer][idx].unaryExpr(se2_grid::Clamp<float>(lowerValue, upperValue));
                const Eigen::MatrixXf& data = map[layer][idx];

                for (se2_grid::SE2GridIterator iterator(map); !iterator.isPastEnd(); ++iterator)
                {
                    Eigen::Array3i index(*iterator);
                    index(2) = idx;
                    const float& value = data(index(0), index(1));
                    if (std::isfinite(value))
                    {
                        const unsigned char imageValue = (unsigned char)(((value - lowerValue) / (upperValue - lowerValue)) * (float)imageMax);
                        const Eigen::Array3i imageIndex(iterator.getUnwrappedIndex());
                        image.at<cv::Vec<unsigned char, 1>>(imageIndex(0), imageIndex(1))[idx] = imageValue;
                    }
                }

                return true;
            }

            bool toImage(const se2_grid::SE2Grid& se2Grid, const std::string& layer, 
                        const size_t idx, const int encoding, cv::Mat& image)
            {
                const float minValue = se2Grid[layer][idx].minCoeffOfFinites();
                const float maxValue = se2Grid[layer][idx].maxCoeffOfFinites();
                return toImage(se2Grid, layer, idx, encoding, minValue, maxValue, image);
            }

        private:

            double radius_ = 5.0;

            std::string inputLayer_;

            std::string outputLayer_;
    };

}  // namespace terrain_analyzer