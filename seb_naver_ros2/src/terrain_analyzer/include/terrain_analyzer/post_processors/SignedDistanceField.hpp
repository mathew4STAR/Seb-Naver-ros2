#pragma once

#include <string>

#include <Eigen/Dense>

#include <se2_grid_core/SE2Grid.hpp>
#include <se2_grid_core/iterators/iterators.hpp>

#include "terrain_analyzer/post_processors/PostProcessorBase.hpp"

namespace terrain_analyzer
{
    template <typename F_get_val, typename F_set_val>
    void fillESDF(F_get_val f_get_val, F_set_val f_set_val, int start, int end, int size) 
    {
        int v[size];
        float z[size + 1];

        int k = start;
        v[start] = start;
        z[start] = -std::numeric_limits<float>::max();
        z[start + 1] = std::numeric_limits<float>::max();

        for (int q = start + 1; q <= end; q++) {
            k++;
            float s;

            do {
            k--;
            s = ((f_get_val(q) + q * q) - (f_get_val(v[k]) + v[k] * v[k])) / (2 * q - 2 * v[k]);
            } while (s <= z[k]);

            k++;

            v[k] = q;
            z[k] = s;
            z[k + 1] = std::numeric_limits<float>::max();
        }

        k = start;

        for (int q = start; q <= end; q++) {
            while (z[k + 1] < q) k++;
            float val = (q - v[k]) * (q - v[k]) + f_get_val(v[k]);
            f_set_val(q, val);
        }
    }

    class SignedDistanceField : public terrain_analyzer::PostProcessor<se2_grid::SE2Grid>
    {
        public:

            SignedDistanceField() {}

            ~SignedDistanceField() override = default;

            bool configure() override
            {
                if (!PostProcessor::getParam(std::string("threshold"), threshold_))
                {
                    RCLCPP_ERROR(rclcpp::get_logger(\"PostProcessors\"), "SignedDistanceField did not find parameter `threshold`.");
                    return false;
                }

                if (!PostProcessor::getParam(std::string("input_layer"), inputLayer_))
                {
                    RCLCPP_ERROR(rclcpp::get_logger(\"PostProcessors\"), "SignedDistanceField did not find parameter `input_layer`.");
                    return false;
                }

                RCLCPP_DEBUG(rclcpp::get_logger(\"PostProcessors\"), "SignedDistanceField input layer is = %s.", inputLayer_.c_str());

                if (!PostProcessor::getParam(std::string("output_layer"), outputLayer_))
                {
                    RCLCPP_ERROR(rclcpp::get_logger(\"PostProcessors\"), "Step filter did not find parameter `output_layer`.");
                    return false;
                }

                RCLCPP_DEBUG(rclcpp::get_logger(\"PostProcessors\"), "SignedDistanceField output_layer = %s.", outputLayer_.c_str());

                return true;
            }

            bool process(const se2_grid::SE2Grid& mapIn, se2_grid::SE2Grid& mapOut) override
            {
                // TODO: improve code logic to eliminate this constraint
                if (!mapIn.isDefaultStartIndex())
                    throw std::runtime_error("SignedDistanceField cannot be used with maps that don't have a default buffer start index.");

                mapOut = mapIn;
                bool has_so2 = mapIn.hasSO2(inputLayer_);
                // mapOut.add(std::string(outputLayer_), has_so2);
                mapOut.add(std::string(outputLayer_), false);

                if (has_so2)
                {
                    // TODO: to be optimized or revise
                    int yaw_size = mapOut.getSizeYaw();
                    Eigen::MatrixXf mat = mapIn[inputLayer_][0];
                    Eigen::MatrixXi mati;
                    mati.resize(mat.rows(), mat.cols());
                    mati.setZero();
                    for (int l=0; l<yaw_size; l++)
                    {
                        mat = mapIn[inputLayer_][l];
                        for (int i=0; i<mat.rows(); i++)
                            for (int j=0; j<mat.cols(); j++)
                            {
                                if (isnan(mat(i, j)))
                                    continue;
                                if (mat(i, j) > threshold_)
                                    mati(i, j) = 1;
                            }
                    }
                        
                    Eigen::MatrixXf res;
                    updateESDF2d(mati, res, mapOut.getResolutionPos(), mapOut.getSizePos()[0]);

                    // for (int i=0; i<yaw_size; i++)
                    // {
                    //     mapOut[outputLayer_][i] = res;
                    // }
                    mapOut[outputLayer_][0] = res;
                }
                else
                {
                    Eigen::MatrixXf mat = mapIn[inputLayer_][0];
                    Eigen::MatrixXi mati;
                    mati.resize(mat.rows(), mat.cols());
                    mati.setZero();
                    for (int i=0; i<mat.rows(); i++)
                        for (int j=0; j<mat.cols(); j++)
                        {
                            if (mat(i, j) > threshold_)
                                mati(i, j) = 1;
                        }
                    Eigen::MatrixXf res;
                    updateESDF2d(mati, res, mapOut.getResolutionPos(), mapOut.getSizePos()[0]);
                    mapOut[outputLayer_][0] = res;
                }
                
                return true;
            }

            void updateESDF2d(const Eigen::MatrixXi& mat, Eigen::MatrixXf& res, double resolution, int size)
            {
                Eigen::Vector2i min_esdf(0, 0);
                Eigen::Vector2i max_esdf(mat.rows()-1, mat.cols()-1);
                int rows = mat.rows();
                int cols = mat.cols();

                Eigen::MatrixXf tmp_buffer1;
                Eigen::MatrixXf tmp_buffer2;
                Eigen::MatrixXf neg_buffer;
                Eigen::MatrixXi neg_map;
                Eigen::MatrixXf dist_buffer;
                tmp_buffer1.resize(rows, cols);
                tmp_buffer2.resize(rows, cols);
                neg_buffer.resize(rows, cols);
                neg_map.resize(rows, cols);
                dist_buffer.resize(rows, cols);
                res.resize(rows, cols);

                /* ========== compute positive DT ========== */

                for (int x = min_esdf[0]; x <= max_esdf[0]; x++)
                {
                    fillESDF(
                        [&](int y)
                        {
                            return mat(x, y) == 1 ?
                                0 :
                                std::numeric_limits<float>::max();
                        },
                        [&](int y, float val) { tmp_buffer1(x, y) = val; }, min_esdf[1],
                        max_esdf[1], size
                    );
                }

                for (int y = min_esdf[1]; y <= max_esdf[1]; y++) {
                    fillESDF(
                        [&](int x) { return tmp_buffer1(x, y); },
                        [&](int x, float val)
                        {
                            dist_buffer(x, y) = resolution * std::sqrt(val);
                        },
                        min_esdf[0], max_esdf[0], size
                    );
                }

                /* ========== compute negative distance ========== */
                for (int x = min_esdf(0); x <= max_esdf(0); ++x)
                    for (int y = min_esdf(1); y <= max_esdf(1); ++y)
                    {
                        if (mat(x, y) == 0)
                        {
                            neg_map(x, y) = 1;
                        } else if (mat(x, y) == 1)
                        {
                            neg_map(x, y) = 0;
                        } else 
                        {
                            RCLCPP_ERROR(rclcpp::get_logger(\"PostProcessors\"), "what?");
                        }
                    }

                for (int x = min_esdf[0]; x <= max_esdf[0]; x++) {
                    fillESDF(
                        [&](int y)
                        {
                            return neg_map(x, y) == 1 ?
                                0 :
                                std::numeric_limits<float>::max();
                        },
                        [&](int y, float val) { tmp_buffer1(x, y) = val; }, min_esdf[1],
                        max_esdf[1], size
                    );
                }

                for (int y = min_esdf[1]; y <= max_esdf[1]; y++)
                {
                    fillESDF(
                        [&](int x) { return tmp_buffer1(x, y); },
                        [&](int x, float val)
                        {
                            neg_buffer(x, y) = resolution * std::sqrt(val);
                        },
                        min_esdf[0], max_esdf[0], size
                    );
                }

                /* ========== combine pos and neg DT ========== */
                for (int x = min_esdf(0); x <= max_esdf(0); ++x)
                    for (int y = min_esdf(1); y <= max_esdf(1); ++y)
                    {
                        res(x, y) = dist_buffer(x, y);
                        if (neg_buffer(x, y) > 0.0)
                            res(x, y) += (-neg_buffer(x, y) + resolution);
                    }
                
                return;
            }

        private:

            double threshold_ = 0.0;

            std::string inputLayer_;

            std::string outputLayer_;
    };

}  // namespace terrain_analyzer