#include "terrain_analyzer/TerrainAnalyzer.hpp"

#ifdef ENABLE_CUDA
using namespace Eigen;
// #define ENABLE_CUDA 1

void gpuInit(const Vector2i& size_pos_,
            int   size_yaw_,
            const Vector2f& length_pos_,
            float resolution_pos_,
            float resolution_yaw_,
            const Vector2f& position_,
            float min_var_,
            float max_var_,
            float hori_var_,
            float mahalanobis_threshold_,
            float normal_radius_,
            float min_cosxi_,
            float max_curvature_,
            float weight_xi_,
            float weight_curvature_,
            float ignore_z_min_,
            float ignore_z_max_,
            float beam_sigma2_,
            const Vector3f& body2sensor_T_);

void computerMapR2(const int& point_num_,
                    float* points_world,
                    const float& resolution_pos_,
                    const Vector2i& size_pos_,
                    const Vector3f& body_T_,
                    const Matrix3f& body_R_,
                    const Matrix4f& body_Cov_ZR_,
                    MatrixXf& elevation_,
                    MatrixXf& var_,
                    MatrixXf& inpainted_,
                    MatrixXf& smooth_,
                    MatrixXf& normal_x_,
                    MatrixXf& normal_y_,
                    MatrixXf& risk_,
                    MatrixXf& sdf_,
                    Array2i& start_idx_,
                    Vector2f& position_,
                    bool verbose_);
#endif

namespace terrain_analyzer
{
#ifdef ENABLE_CUDA
    bool TerrainAnalyzer::init(ros::NodeHandle& nh)
    {
        std::vector<float> sensor2BodyT;
        Vector3f body2sensor_T;
        float ignore_z_min;
        float ignore_z_max;
        float beam_sigma2;
        float length_pos[2];
        float resolution_pos;
        float resolution_yaw;
        float min_var;
        float max_var;
        float hori_var;
        float mahalanobis_threshold;
        float normal_radius;
        float min_cosxi;
        float max_curvature;
        float weight_xi;
        float weight_curvature;

        nh.getParam("/terrain_analyzer/sensor_processors/ignore_z_max", ignore_z_max);
        nh.getParam("/terrain_analyzer/sensor_processors/ignore_z_min", ignore_z_min);
        nh.getParam("/terrain_analyzer/sensor_processors/ouster/beam_sigma", beam_sigma2);
        nh.param<std::vector<float>>("/terrain_analyzer/sensor_processors/sensor2BodyT", sensor2BodyT, std::vector<float>());
        nh.getParam("/terrain_analyzer/elevation_map/length_pos_x", length_pos[0]);
        nh.getParam("/terrain_analyzer/elevation_map/length_pos_y", length_pos[1]);
        nh.getParam("/terrain_analyzer/elevation_map/resolution_pos", resolution_pos);
        nh.getParam("/terrain_analyzer/elevation_map/resolution_yaw", resolution_yaw);
        nh.getParam("/terrain_analyzer/elevation_map/min_var", min_var);
        nh.getParam("/terrain_analyzer/elevation_map/max_var", max_var);
        nh.getParam("/terrain_analyzer/elevation_map/min_var_hori", hori_var);
        nh.getParam("/terrain_analyzer/elevation_map/mahalanobis_threshold", mahalanobis_threshold);
        nh.getParam("/terrain_analyzer/gpu_added_params/normal_radius", normal_radius);
        nh.getParam("/terrain_analyzer/gpu_added_params/min_cosxi", min_cosxi);
        nh.getParam("/terrain_analyzer/gpu_added_params/max_curvature", max_curvature);
        nh.getParam("/terrain_analyzer/gpu_added_params/weight_xi", weight_xi);
        nh.getParam("/terrain_analyzer/gpu_added_params/weight_curvature", weight_curvature);
        nh.getParam("/terrain_analyzer/gpu_added_params/verbose", verbose);

        if (!post_processors.configure("/terrain_analyzer/post_processors", nh))
            return false;

        resolution_yaw = 8.0;
        
        beam_sigma2 *= beam_sigma2;
        for (size_t i=0; i<3; i++)
        {
            body2sensor_T(i) = -sensor2BodyT[i];
        }

        Array2d length_pos_;
        length_pos_[0] = length_pos[0];
        length_pos_[1] = length_pos[1];
        double resolution_pos_ = resolution_pos;
        double resolution_yaw_ = resolution_yaw;
        fused_map.initGeometry(length_pos_, resolution_pos_, resolution_yaw_, Vector2d::Zero());
        sdf_map.initGeometry(length_pos_, resolution_pos_, 8.0, Vector2d::Zero());

        Array2i size_pos_ = fused_map.getSizePos();
        int size_pos[2];
        size_pos[0] = size_pos_(0);
        size_pos[1] = size_pos_(1);
        int size_yaw = fused_map.getSizeYaw();
        float position[2] = {0.0f, 0.0f};

        gpuInit(size_pos_.matrix(),
                size_yaw,
                length_pos_.matrix().cast<float>(),
                resolution_pos,
                resolution_yaw,
                fused_map.getPosition().cast<float>(),
                min_var,
                max_var,
                hori_var,
                mahalanobis_threshold,
                normal_radius,
                min_cosxi,
                max_curvature,
                weight_xi,
                weight_curvature,
                ignore_z_min,
                ignore_z_max,
                beam_sigma2,
                body2sensor_T);

        fused_pub = nh.advertise<se2_grid_msgs::SE2Grid>("fused_map", 1);
        sdf_pub = nh.advertise<se2_grid_msgs::SE2Grid>("sdf_map", 1);
        normal_pub = nh.advertise<visualization_msgs::Marker>("normal_map", 1);
        cloud_sub = nh.subscribe<sensor_msgs::PointCloud2>("cloud", 1, &TerrainAnalyzer::cloudCallback, this);

        odom_sub.subscribe(nh, "odom", 1);
        odom_cache.connectInput(odom_sub);
        odom_cache.setCacheSize(cache_size);
        map_timer = nh.createTimer(ros::Duration(0.5), &TerrainAnalyzer::mapCallback, this);

        return true;
    }

    void TerrainAnalyzer::cloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
    {
        clock_t t0, t1;

        t0 = clock();
        // get datas
        pcl::PointCloud<pcl::PointXYZ> pointCloud;
        pcl::fromROSMsg(*msg, pointCloud);

        Matrix<double, 6, 6> robotPoseCovariance;
        robotPoseCovariance.setZero();
        
        boost::shared_ptr<nav_msgs::Odometry const> poseMessage = odom_cache.getElemBeforeTime(ros::Time::now());
        if (!poseMessage) 
        {
            ROS_WARN("Odometry time is error, continue.");
            return;
        }
        robotPoseCovariance = Map<const MatrixXd>(poseMessage->pose.covariance.data(), 6, 6);
        // ready to send to GPU
        MatrixXf& elevation = fused_map["elevation"][0];
        MatrixXf& var = fused_map["var"][0];
        MatrixXf& inpainted = fused_map["inpainted"][0];
        MatrixXf& smooth = fused_map["smooth"][0];
        MatrixXf& normal_x = fused_map["normal_x"][0];
        MatrixXf& normal_y = fused_map["normal_y"][0];
        MatrixXf& risk = fused_map["risk"][0];
        MatrixXf& sdf = sdf_map["sdf"][0];

        int point_num = pointCloud.size();
        float points_world[point_num*3];
        for (size_t i=0; i<point_num; i++)
        {
            points_world[i*3] = pointCloud.points[i].x;
            points_world[i*3+1] = pointCloud.points[i].y;
            points_world[i*3+2] = pointCloud.points[i].z;
        }
        
        Vector3f body_T = Vector3f(poseMessage->pose.pose.position.x,
                                    poseMessage->pose.pose.position.y,
                                    poseMessage->pose.pose.position.z);
        Matrix3f body_R = Quaternionf(poseMessage->pose.pose.orientation.w, 
                                    poseMessage->pose.pose.orientation.x,
                                    poseMessage->pose.pose.orientation.y,
                                    poseMessage->pose.pose.orientation.z).matrix();
        Matrix4f body_Cov_ZR = robotPoseCovariance.bottomRightCorner(4, 4).cast<float>();

        Vector2f position_map = fused_map.getPosition().cast<float>();
        Array2i start_idx_map = fused_map.getStartIndex();
        Vector2i size_pose_map = fused_map.getSizePos().matrix();
        float resolution_pos = fused_map.getResolutionPos();

        computerMapR2(point_num,
                    points_world,
                    resolution_pos,
                    size_pose_map,
                    body_T,
                    body_R,
                    body_Cov_ZR,
                    elevation,
                    var,
                    inpainted,
                    smooth,
                    normal_x,
                    normal_y,
                    risk,
                    sdf,
                    start_idx_map,
                    position_map,
                    verbose);
        
        if (verbose)
        {
            fused_map.setPosition(position_map.cast<double>());
            fused_map.setStartIndex(start_idx_map);
        }

        sdf_map.setPosition(position_map.cast<double>());
        sdf_map.setStartIndex(start_idx_map);
        sdf_map.convertToDefaultStartIndex();
        post_processors.process(sdf_map, sdf_map);

        t1 = clock();
        if ((double)(t1 - t0)/1000.0 > 100.0)
            ROS_WARN("Mapping time consuming = %fms", (double)(t1 - t0)/1000.0);

        last_update_time = ros::Time::now();

        return;
    }

    void TerrainAnalyzer::mapCallback(const ros::TimerEvent& /*event*/)
    {
        se2_grid_msgs::SE2Grid map_msg, sdf_msg;
        visualization_msgs::Marker normal_msg;

        se2_grid::SE2GridRosConverter::toMessage(sdf_map, sdf_msg);
        sdf_pub.publish(sdf_msg);
        if (!verbose)
            return;

        se2_grid::SE2GridRosConverter::toMessage(fused_map, map_msg);
        normal_msg.type = visualization_msgs::Marker::LINE_LIST;
        normal_msg.header.frame_id = "map";
        normal_msg.pose.orientation.w = 1.0;
        normal_msg.scale.x = 0.006;
        normal_msg.color.a = 0.6;
        normal_msg.color.r = 1.0;
        normal_msg.color.g = 1.0;
        normal_msg.color.b = 1.0;
        geometry_msgs::Point p1, p2;
        
        int yaw_idx = 0;
        se2_grid::SE2Grid map_copy = fused_map;
        double res_pos = map_copy.getResolutionPos();
        std::vector<std::string> normal_related;
        std::string normal_x("normal_x");
        std::string normal_y("normal_y");
        std::string elevation_layer("elevation");
        elevation_layer = std::string("inpainted");
        normal_related.emplace_back(normal_x);
        normal_related.emplace_back(normal_y);
        normal_related.emplace_back(elevation_layer);

        for (se2_grid::SE2GridIterator iterator(map_copy); !iterator.isPastEnd(); ++iterator) 
        {
            Vector3d sub_position;
            Array3i  sub_idx = *iterator;
            if (sub_idx(0)%2==0 && sub_idx(1)%2==0)
            {
                sub_idx(2) = yaw_idx;
                map_copy.index2Pos(sub_idx, sub_position);
                if (!map_copy.isValid(sub_idx, normal_related))
                    continue;
                double zbx = map_copy.at(normal_x, sub_idx);
                double zby = map_copy.at(normal_y, sub_idx);
                p1.x = sub_position.x();
                p1.y = sub_position.y();
                p1.z = map_copy.at(elevation_layer, sub_idx);
                p2.x = p1.x + 1.5 * res_pos * zbx;
                p2.y = p1.y + 1.5 * res_pos * zby;
                p2.z = p1.z + 1.5 * res_pos * sqrt(1.0-zbx*zbx-zby*zby);
                normal_msg.points.emplace_back(p1);
                normal_msg.points.emplace_back(p2);
            }
        }

        fused_pub.publish(map_msg);
        if (!normal_msg.points.empty())
            normal_pub.publish(normal_msg);

        return;
    }

#else
    bool TerrainAnalyzer::init(ros::NodeHandle& nh)
    {
        if (!sensor_processor.init(nh))
        {
            ROS_ERROR("Sensor Processor init ERROR!");
            return false;
        }

        Eigen::Array2d len_pos;
        double resolution_pos;
        double resolution_yaw;

        nh.getParam("elevation_map/length_pos_x", len_pos(0));
        nh.getParam("elevation_map/length_pos_y", len_pos(1));
        nh.getParam("elevation_map/resolution_pos", resolution_pos);
        nh.getParam("elevation_map/min_var", min_var);
        nh.getParam("elevation_map/max_var", max_var);
        nh.getParam("elevation_map/min_var_hori", min_var_hori);
        nh.getParam("elevation_map/max_var_hori", max_var_hori);
        nh.getParam("elevation_map/mahalanobis_threshold", mahalanobis_threshold);
        
        if (!post_processors.configure("post_processors", nh))
            return false;
        
        raw_map.initGeometry(len_pos, resolution_pos, 6.0);
        fused_map.initGeometry(len_pos, resolution_pos, 6.0);

        raw_pub = nh.advertise<se2_grid_msgs::SE2Grid>("/raw_map", 1);
        fused_pub = nh.advertise<se2_grid_msgs::SE2Grid>("/fused_map", 1);
        normal_pub = nh.advertise<visualization_msgs::Marker>("/normal_map", 1);
        // shit_pub = nh.advertise<sensor_msgs::PointCloud2>("/pc_shit", 1);

        cloud_sub = nh.subscribe<sensor_msgs::PointCloud2>("cloud", 1, &TerrainAnalyzer::cloudCallback, this);

        map_timer = nh.createTimer(ros::Duration(0.5), &TerrainAnalyzer::mapCallback, this);

        odom_sub.subscribe(nh, "odom", 1);
        odom_cache.connectInput(odom_sub);
        odom_cache.setCacheSize(cache_size);

        return true;
    }

    void TerrainAnalyzer::cloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
    {
        ros::Time t0 = ros::Time::now();

        pcl::PointCloud<pcl::PointXYZ> pointCloud;
        pcl::fromROSMsg(*msg, pointCloud);

        Eigen::Matrix<double, 6, 6> robotPoseCovariance;
        robotPoseCovariance.setZero();
        
        boost::shared_ptr<nav_msgs::Odometry const> poseMessage = odom_cache.getElemBeforeTime(ros::Time::now());
        if (!poseMessage) 
        {
            ROS_WARN("Odometry time is error, continue.");
            return;
        }
        robotPoseCovariance = Eigen::Map<const Eigen::MatrixXd>(poseMessage->pose.covariance.data(), 6, 6);

        // Process point cloud.
        pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloudProcessed(new pcl::PointCloud<pcl::PointXYZ>);
        Eigen::VectorXf measureVars;
        Eigen::Matrix4d bodyTrans = Eigen::Matrix4d::Identity();
        bodyTrans.topLeftCorner(3, 3) = Eigen::Quaterniond(poseMessage->pose.pose.orientation.w, 
                                                            poseMessage->pose.pose.orientation.x,
                                                            poseMessage->pose.pose.orientation.y,
                                                            poseMessage->pose.pose.orientation.z).matrix();
        bodyTrans.topRightCorner(3, 1) = Eigen::Vector3d(poseMessage->pose.pose.position.x,
                                                        poseMessage->pose.pose.position.y,
                                                        poseMessage->pose.pose.position.z);
        sensor_processor.process(pointCloud, bodyTrans, robotPoseCovariance, 
                                pointCloudProcessed, measureVars);
        // Move and add
        raw_map.move(bodyTrans.topRightCorner(2, 1));
        add(pointCloudProcessed, measureVars);
        raw_map.convertToDefaultStartIndex();
        post_processors.process(raw_map, raw_map);
        fuseAll();

        last_update_time = ros::Time::now();

        return;
    }

    void TerrainAnalyzer::mapCallback(const ros::TimerEvent& /*event*/)
    {
        se2_grid_msgs::SE2Grid raw_msg, fused_msg;
        visualization_msgs::Marker normal_msg;

        se2_grid::SE2GridRosConverter::toMessage(raw_map, raw_msg);
        se2_grid::SE2GridRosConverter::toMessage(fused_map, fused_msg);

        normal_msg.type = visualization_msgs::Marker::LINE_LIST;
        normal_msg.header.frame_id = "map";
        normal_msg.pose.orientation.w = 1.0;
        normal_msg.scale.x = 0.006;
        normal_msg.color.a = 0.6;
        normal_msg.color.r = 1.0;
        normal_msg.color.g = 1.0;
        normal_msg.color.b = 1.0;
        geometry_msgs::Point p1, p2;
        
        int yaw_idx = 0;
        std::vector<std::string> normal_related;
        if (!raw_map.exists("normal_vectors_x"))
            return;
        double res_pos = raw_map.getResolutionPos();
        normal_related.emplace_back("elevation_inpainted");
        normal_related.emplace_back("normal_vectors_x");
        normal_related.emplace_back("normal_vectors_y");
        for (se2_grid::SE2GridIterator iterator(raw_map); !iterator.isPastEnd(); ++iterator) 
        {
            Eigen::Vector3d sub_position;
            Eigen::Array3i  sub_idx = *iterator;
            if (sub_idx(0)%2==0 && sub_idx(1)%2==0)
            {
                sub_idx(2) = yaw_idx;
                raw_map.index2Pos(sub_idx, sub_position);
                if (!raw_map.isValid(sub_idx, normal_related))
                    continue;
                double zbx = raw_map.at("normal_vectors_x", sub_idx);
                double zby = raw_map.at("normal_vectors_y", sub_idx);
                p1.x = sub_position.x();
                p1.y = sub_position.y();
                p1.z = raw_map.at("elevation_inpainted", sub_idx);
                p2.x = p1.x + 1.5 * res_pos * zbx;
                p2.y = p1.y + 1.5 * res_pos * zby;
                p2.z = p1.z + 1.5 * res_pos * sqrt(1.0-zbx*zbx-zby*zby);
                normal_msg.points.emplace_back(p1);
                normal_msg.points.emplace_back(p2);
            }
        }

        raw_pub.publish(raw_msg);
        fused_pub.publish(fused_msg);
        if (!normal_msg.points.empty())
            normal_pub.publish(normal_msg);

        return;
    }
#endif

    bool TerrainAnalyzer::add(const pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloudMap, Eigen::VectorXf& pointVariances)
    {
        auto& elevationLayer = raw_map["elevation"][0];
        auto& varianceLayer = raw_map["var"][0];
        auto& varXLayer = raw_map["var_x"][0];
        auto& varYLayer = raw_map["var_y"][0];
        auto& varXYLayer = raw_map["var_xy"][0];

        for (unsigned int i = 0; i < pointCloudMap->points.size(); ++i) 
        {
            auto& point = pointCloudMap->points[i];
            Eigen::Array3i index;
            Eigen::Vector3d position(point.x, point.y, 0);
            
            if (!raw_map.pos2Index(position, index))
                continue;

            auto& elevation = elevationLayer(index(0), index(1));
            auto& variance = varianceLayer(index(0), index(1));
            auto& varX = varXLayer(index(0), index(1));
            auto& varY = varYLayer(index(0), index(1));
            auto& varXY = varXYLayer(index(0), index(1));
            float pointVariance = std::max(fabs(pointVariances(i)), (float)min_var_hori);

            if (!raw_map.isValid(index, raw_basic_layers))
            {
                elevation = point.z;
                variance = pointVariance;
                varX = min_var_hori;
                varY = min_var_hori;
                varXY = 0.0;
                continue;
            }

            const double mahalanobisDistance = fabs(point.z - elevation) / sqrt(variance); 
            if (mahalanobisDistance > mahalanobis_threshold)
            {
                //TODO: deal with multiple heights in one cell.
                continue;
            }

            // Fuse measurement with elevation map data. Using Kalman filtering
            elevation = (variance * point.z + pointVariance * elevation) / (variance + pointVariance); 
            variance = (pointVariance * variance) / (pointVariance + variance);

            varX = min_var_hori;
            varY = min_var_hori;
            varXY = 0.0;
        }

        raw_map["var"][0] = raw_map["var"][0].unaryExpr(ClampVar(min_var, max_var));
        raw_map["var_x"][0] = raw_map["var_x"][0].unaryExpr(ClampVar(min_var_hori, max_var_hori));
        raw_map["var_y"][0] = raw_map["var_y"][0].unaryExpr(ClampVar(min_var_hori, max_var_hori));

        return true;
    }

    bool TerrainAnalyzer::fuseAll()
    {
        auto raw_copy = raw_map;

        const double half_res_pos = fused_map.getResolutionPos() / 2.0;
        const float min_weight = std::numeric_limits<float>::epsilon() * static_cast<float>(2.0);
        
        const double ellipse_extension = M_SQRT2 * fused_map.getResolutionPos();

        if (raw_copy.getPosition() != fused_map.getPosition())
        {
            fused_map.move(raw_copy.getPosition());
            fused_map.convertToDefaultStartIndex();
        }

        for (se2_grid::SE2GridIterator iterator(raw_copy); !iterator.isPastEnd(); ++iterator) 
        {
            if (fused_map.isValid(*iterator, fused_basic_layers))
                continue;

            if (!raw_copy.isValid(*iterator, raw_basic_layers))
                continue;

            // Get size of error ellipse.
            const float& sigmaXsquare = raw_copy.at("var_x", *iterator);
            const float& sigmaYsquare = raw_copy.at("var_y", *iterator);
            const float& sigmaXYsquare = raw_copy.at("var_xy", *iterator);

            Eigen::Matrix2d covarianceMatrix;
            covarianceMatrix << sigmaXsquare, sigmaXYsquare, sigmaXYsquare, sigmaYsquare;
            // 95.45% confidence ellipse which is 2.486-sigma for 2 dof problem.
            // http://www.reid.ai/2012/09/chi-squared-distribution-table-with.html
            const double uncertaintyFactor = 2.486;  // sqrt(6.18)
            Eigen::EigenSolver<Eigen::Matrix2d> solver(covarianceMatrix);
            Eigen::Array2d eigenvalues(solver.eigenvalues().real().cwiseAbs());

            Eigen::Array2d::Index max_eigen_idx{0};
            eigenvalues.maxCoeff(&max_eigen_idx);
            Eigen::Array2d::Index min_eigen_idx{0};
            max_eigen_idx == Eigen::Array2d::Index(0) ? min_eigen_idx = 1 : min_eigen_idx = 0;
            const Eigen::Array2d ellipseLength = 2.0 * uncertaintyFactor 
                                                * Eigen::Array2d(eigenvalues(max_eigen_idx), eigenvalues(min_eigen_idx)).sqrt()
                                                + ellipse_extension;
            const double ellipseRotation(atan2(solver.eigenvectors().col(max_eigen_idx).real()(1), 
                                         solver.eigenvectors().col(max_eigen_idx).real()(0)));

            Eigen::Vector3d sub_position;
            raw_copy.index2Pos(*iterator, sub_position);
            se2_grid::EllipseIterator ellipseIterator(raw_copy, sub_position.head(2), ellipseLength, ellipseRotation);

            Eigen::ArrayXf means;
            Eigen::ArrayXf weights;
            const unsigned int maxNumberOfCellsToFuse = ellipseIterator.getSubmapSize().prod();
            means.resize(maxNumberOfCellsToFuse);
            weights.resize(maxNumberOfCellsToFuse);
            WeightedEmpiricalCumulativeDistributionFunction<float> lowerBoundDistribution;
            WeightedEmpiricalCumulativeDistributionFunction<float> upperBoundDistribution;

            float maxStandardDeviation = sqrt(eigenvalues(max_eigen_idx));
            float minStandardDeviation = sqrt(eigenvalues(min_eigen_idx));
            Eigen::Rotation2Dd rotationMatrix(ellipseRotation);

            size_t i = 0;
            for (; !ellipseIterator.isPastEnd(); ++ellipseIterator)
            {
                if (!raw_copy.isValid(*ellipseIterator, raw_basic_layers))
                    continue;

                means[i] = raw_copy.at("elevation", *ellipseIterator);

                // Compute weight from probability.
                Eigen::Vector3d absolutePosition;
                raw_copy.index2Pos(*ellipseIterator, absolutePosition);
                Eigen::Vector2d distanceToCenter = (rotationMatrix * (absolutePosition.head(2) - sub_position)).cwiseAbs();

                float probability1 = cumulativeDistributionFunction(distanceToCenter.x() + half_res_pos, 0.0, maxStandardDeviation) -
                                    cumulativeDistributionFunction(distanceToCenter.x() - half_res_pos, 0.0, maxStandardDeviation);
                float probability2 = cumulativeDistributionFunction(distanceToCenter.y() + half_res_pos, 0.0, minStandardDeviation) -
                                    cumulativeDistributionFunction(distanceToCenter.y() - half_res_pos, 0.0, minStandardDeviation);

                const float weight = std::max(min_weight, probability1 * probability2);
                weights[i] = weight;
                const float standardDeviation = sqrt(raw_copy.at("var", *ellipseIterator));
                lowerBoundDistribution.add(means[i] - 2.0 * standardDeviation, weight);
                upperBoundDistribution.add(means[i] + 2.0 * standardDeviation, weight);

                i++;
            }

            if (i == 0)
            {
                // Nothing to fuse.
                fused_map.at("elevation", *iterator) = raw_copy.at("elevation", *iterator);
                fused_map.at("lower_bound", *iterator) =
                    raw_copy.at("elevation", *iterator) - 2.0 * sqrt(raw_copy.at("var", *iterator));
                fused_map.at("upper_bound", *iterator) =
                    raw_copy.at("elevation", *iterator) + 2.0 * sqrt(raw_copy.at("var", *iterator));
                continue;
            }

            // Fuse.
            means.conservativeResize(i);
            weights.conservativeResize(i);

            float mean = (weights * means).sum() / weights.sum();

            if (!std::isfinite(mean))
            {
                ROS_ERROR("Something went wrong when fusing the map: Mean = %f", mean);
                continue;
            }

            // Add to fused map.
            fused_map.at("elevation", *iterator) = mean;
            lowerBoundDistribution.compute();
            upperBoundDistribution.compute();
            fused_map.at("lower_bound", *iterator) = lowerBoundDistribution.quantile(0.01);
            fused_map.at("upper_bound", *iterator) = upperBoundDistribution.quantile(0.99);
        }
        return true;
    }
}