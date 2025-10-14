#include "super_grasp/superquadric.h"
#include <ros/ros.h>
#include <chrono>
#include <random>
#include <cmath>

namespace super_grasp {

Superquadric::Superquadric(std::shared_ptr<open3d::geometry::PointCloud> pcd,
                           int downsample,
                           bool debug)
    : downsample_(downsample), debug_(debug) {

    auto init_time = std::chrono::high_resolution_clock::now();

    // Estimate shape parameters
    auto [e1, e2] = estimateE(pcd);
    e1_ = e1;
    e2_ = e2;

    // Create superquadric mesh and pose
    auto [sq, pose] = createSuperquadric(pcd, e1_, e2_);
    superquadric_ = sq;
    pose_ = pose;

    if (debug_) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - init_time);
        ROS_INFO("Superquadric: time: %.3fs", duration.count() / 1000.0);
        ROS_INFO("Superquadric: Num Points: %zu", superquadric_->points_.size());
    }
}

Eigen::Vector3d Superquadric::calculateKurtosis(const Eigen::MatrixXd& points) {
    // Calculate Fisher kurtosis for each axis
    // Kurtosis = E[(X-μ)^4] / σ^4 - 3 (Fisher/excess kurtosis)
    
    Eigen::Vector3d kurtosis;
    int n = points.rows();
    
    for (int axis = 0; axis < 3; ++axis) {
        Eigen::VectorXd col = points.col(axis);
        
        double mean = col.mean();
        double variance = (col.array() - mean).square().sum() / n;
        double std_dev = std::sqrt(variance);
        
        if (std_dev < 1e-10) {
            kurtosis(axis) = 0.0;
            continue;
        }
        
        // Fourth moment
        double fourth_moment = (col.array() - mean).pow(4).sum() / n;
        
        // Fisher kurtosis (excess kurtosis)
        kurtosis(axis) = (fourth_moment / (variance * variance)) - 3.0;
    }
    
    return kurtosis;
}

std::pair<double, double> Superquadric::estimateE(
    std::shared_ptr<open3d::geometry::PointCloud> pcd) {
    
    try {
        Eigen::MatrixXd points(pcd->points_.size(), 3);
        for (size_t i = 0; i < pcd->points_.size(); ++i) {
            points.row(i) = pcd->points_[i];
        }

        Eigen::Vector3d centroid = pcd->GetCenter();
        Eigen::MatrixXd centered_points = points.rowwise() - centroid.transpose();

        // Compute kurtosis
        Eigen::Vector3d krt = calculateKurtosis(centered_points);

        // Shape parameter along z-axis
        double e1 = std::clamp(1.0 + (krt(2) - 3.0) * 0.1, 0.3, 2.0);

        // Shape parameter along xy-plane
        double e2 = std::clamp(1.0 + ((krt(0) + krt(1)) / 2.0 - 3.0) * 0.1, 0.3, 2.0);

        if (debug_) {
            ROS_INFO("Estimated e1: %.3f, e2: %.3f", e1, e2);
        }

        return {e1, e2};

    } catch (const std::exception& e) {
        ROS_ERROR("superquadric [estimateE] Error: %s", e.what());
        return {1.0, 1.0};
    }
}

std::pair<std::shared_ptr<open3d::geometry::PointCloud>, Eigen::Matrix3d>
Superquadric::createSuperquadric(std::shared_ptr<open3d::geometry::PointCloud> pcd,
                                 double e1, double e2) {
    try {
        // Get point cloud data
        Eigen::MatrixXd points(pcd->points_.size(), 3);
        for (size_t i = 0; i < pcd->points_.size(); ++i) {
            points.row(i) = pcd->points_[i];
        }

        // Check if point cloud is too flat
        Eigen::Vector3d min_pt = points.colwise().minCoeff();
        Eigen::Vector3d max_pt = points.colwise().maxCoeff();
        Eigen::Vector3d ranges = max_pt - min_pt;

        if (debug_) {
            ROS_INFO("Point cloud ranges: X=%.4f, Y=%.4f, Z=%.4f", 
                     ranges(0), ranges(1), ranges(2));
        }

        // If any dimension is too small, add artificial thickness
        double min_range = 0.01; // 1cm
        for (int dim = 0; dim < 3; ++dim) {
            if (ranges(dim) < min_range) {
                if (debug_) {
                    ROS_INFO("Segment is flat in dimension %d, adding artificial thickness", dim);
                }
                
                // Add small noise in the flat dimension
                std::random_device rd;
                std::mt19937 gen(rd());
                std::normal_distribution<> dist(0.0, min_range / 4.0);
                
                for (size_t i = 0; i < pcd->points_.size(); ++i) {
                    pcd->points_[i](dim) += dist(gen);
                }
            }
        }

        // Compute oriented bounding box
        auto obb = pcd->GetOrientedBoundingBox();
        double a1 = obb.extent_(0) / 2.0;
        double a2 = obb.extent_(1) / 2.0;
        double a3 = obb.extent_(2) / 2.0;
        Eigen::Matrix3d R_matrix = obb.R_;
        Eigen::Vector3d center = obb.center_;

        // Decide target number of points
        int n_points = std::max(100, static_cast<int>(pcd->points_.size() * downsample_ / 100));

        // Random number generation
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> uniform_dist(-1.0, 1.0);

        // Generate superquadric parameters
        std::vector<double> eta(n_points);
        std::vector<double> omega(n_points);
        
        for (int i = 0; i < n_points; ++i) {
            double u = uniform_dist(gen);
            eta[i] = std::asin(std::clamp(u, -1.0, 1.0));
            omega[i] = uniform_dist(gen) * M_PI;
        }

        // Generate superquadric points
        Eigen::MatrixXd V_local(n_points, 3);
        
        for (int i = 0; i < n_points; ++i) {
            double ce = std::cos(eta[i]);
            double se = std::sin(eta[i]);
            double co = std::cos(omega[i]);
            double so = std::sin(omega[i]);

            double x = a1 * sgn(ce) * std::pow(std::abs(ce), e1) * 
                       sgn(co) * std::pow(std::abs(co), e2);
            double y = a2 * sgn(ce) * std::pow(std::abs(ce), e1) * 
                       sgn(so) * std::pow(std::abs(so), e2);
            double z = a3 * sgn(se) * std::pow(std::abs(se), e1);

            V_local.row(i) << x, y, z;
        }

        // Transform to world coordinates
        Eigen::MatrixXd V_world = (R_matrix * V_local.transpose()).transpose();
        V_world.rowwise() += center.transpose();

        // Build point cloud
        auto sq_pcd = std::make_shared<open3d::geometry::PointCloud>();
        sq_pcd->points_.resize(n_points);
        
        for (int i = 0; i < n_points; ++i) {
            sq_pcd->points_[i] = V_world.row(i);
        }

        // Estimate normals
        sq_pcd->EstimateNormals(
            open3d::geometry::KDTreeSearchParamHybrid(0.02, 30));

        return {sq_pcd, R_matrix};

    } catch (const std::exception& e) {
        ROS_ERROR("superquadric [createSuperquadric] Error: %s", e.what());
        return {std::make_shared<open3d::geometry::PointCloud>(), Eigen::Matrix3d::Identity()};
    }
}

} // namespace super_grasp