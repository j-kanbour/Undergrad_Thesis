#include "super_grasp/grasps.h"
#include <ros/ros.h>
#include <chrono>
#include <algorithm>

namespace super_grasp {

Grasps::Grasps(const std::vector<std::shared_ptr<open3d::geometry::PointCloud>>& sq,
               const std::vector<Eigen::Matrix3d>& sq_poses,
               const std::string& target_frame,
               const std::string& orientation,
               double grasp_width,
               bool debug)
    : target_frame_(target_frame),
      orientation_(orientation),
      grasp_width_(grasp_width),
      debug_(debug) {

    if (debug_) {
        grasp_points_ = std::make_shared<open3d::geometry::PointCloud>();
    }

    auto time_start = std::chrono::high_resolution_clock::now();

    // Filter and select best grasp point
    auto [point, pose] = graspPointFiltering(sq, sq_poses);
    primary_point_ = point;
    sq_pose_ = pose;

    auto time1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(time1 - time_start);

    // Generate pose
    selected_grasps_ = generatePose(primary_point_, target_frame_, sq_pose_);

    auto time2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1);

    if (debug_) {
        ROS_INFO("Grasps: Grasp points generated within %.3fs", duration1.count() / 1000.0);
        ROS_INFO("Grasps: Grasp pose generated within %.3fs", duration2.count() / 1000.0);
    }
}

std::pair<Eigen::Vector3d, Eigen::Matrix3d> Grasps::graspPointFiltering(
    const std::vector<std::shared_ptr<open3d::geometry::PointCloud>>& sq_list,
    const std::vector<Eigen::Matrix3d>& sq_poses) {

    try {
        if (sq_list.empty() || sq_poses.empty()) {
            ROS_WARN("Grasps: Empty superquadric list");
            return {Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity()};
        }

        // Create pairs of superquadrics and poses for sorting
        std::vector<std::pair<std::shared_ptr<open3d::geometry::PointCloud>, Eigen::Matrix3d>> sq_pairs;
        for (size_t i = 0; i < sq_list.size() && i < sq_poses.size(); ++i) {
            sq_pairs.push_back({sq_list[i], sq_poses[i]});
        }

        if (orientation_ == "front" || orientation_.empty()) {
            // Sort by Euclidean distance (closest to camera origin)
            std::sort(sq_pairs.begin(), sq_pairs.end(),
                [](const auto& a, const auto& b) {
                    auto obb_a = a.first->GetOrientedBoundingBox();
                    auto obb_b = b.first->GetOrientedBoundingBox();
                    return obb_a.center_.norm() < obb_b.center_.norm();
                });

            return {sq_pairs[0].first->GetCenter(), sq_pairs[0].second};

        } else if (orientation_ == "top") {
            // Sort by Y-axis (highest object first - lowest Y value in camera frame)
            std::sort(sq_pairs.begin(), sq_pairs.end(),
                [](const auto& a, const auto& b) {
                    auto obb_a = a.first->GetOrientedBoundingBox();
                    auto obb_b = b.first->GetOrientedBoundingBox();
                    return obb_a.center_(1) < obb_b.center_(1);
                });

            return {sq_pairs[0].first->GetCenter(), sq_pairs[0].second};
        }

        return {Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity()};

    } catch (const std::exception& e) {
        ROS_ERROR("grasp [graspPointFiltering] Error: %s", e.what());
        return {Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity()};
    }
}

Eigen::Quaterniond Grasps::rotationMatrixToQuaternion(const Eigen::Matrix3d& R) {
    // Convert rotation matrix to quaternion using Eigen
    Eigen::Quaterniond q(R);
    q.normalize();
    return q;
}

geometry_msgs::PoseStamped Grasps::generatePose(const Eigen::Vector3d& grasp_point,
                                                 const std::string& frame_id,
                                                 const Eigen::Matrix3d& sq_pose) {
    try {
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.frame_id = frame_id;
        pose_stamped.header.stamp = ros::Time::now();

        // Set position
        pose_stamped.pose.position.x = grasp_point(0);
        pose_stamped.pose.position.y = grasp_point(1);
        pose_stamped.pose.position.z = grasp_point(2);

        // Convert rotation matrix to quaternion
        Eigen::Quaterniond scipy_rotation = rotationMatrixToQuaternion(sq_pose);

        if (orientation_ == "front" || orientation_.empty()) {
            // Apply 180° rotation about Y-axis
            Eigen::Quaterniond rotation_y_180(
                Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitY()));
            scipy_rotation = scipy_rotation * rotation_y_180;

        } else { // "top" orientation
            // Apply 90° rotation about X-axis
            Eigen::Quaterniond rotation_x_90(
                Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitX()));
            scipy_rotation = scipy_rotation * rotation_x_90;
        }

        // Normalize and set orientation
        scipy_rotation.normalize();
        pose_stamped.pose.orientation.x = scipy_rotation.x();
        pose_stamped.pose.orientation.y = scipy_rotation.y();
        pose_stamped.pose.orientation.z = scipy_rotation.z();
        pose_stamped.pose.orientation.w = scipy_rotation.w();

        if (debug_) {
            ROS_INFO("grasp: Generated grasp pose at position: [%.3f, %.3f, %.3f]",
                     grasp_point(0), grasp_point(1), grasp_point(2));
            ROS_INFO("grasp: Orientation (quaternion): [%.3f, %.3f, %.3f, %.3f]",
                     scipy_rotation.x(), scipy_rotation.y(), 
                     scipy_rotation.z(), scipy_rotation.w());
        }

        return pose_stamped;

    } catch (const std::exception& e) {
        ROS_ERROR("grasp [generatePose] Error: %s", e.what());
        return geometry_msgs::PoseStamped();
    }
}

} // namespace super_grasp