#ifndef SUPER_GRASP_GRASPS_H
#define SUPER_GRASP_GRASPS_H

#include <open3d/Open3D.h>
#include <Eigen/Dense>
#include <geometry_msgs/PoseStamped.h>
#include <vector>
#include <string>
#include <memory>

namespace super_grasp {

/**
 * @brief Generates and selects grasp poses from superquadric fits
 * 
 * Takes superquadric models and their poses, filters based on orientation,
 * and generates grasp poses for robot manipulation.
 */
class Grasps {
public:
    /**
     * @brief Constructor
     * @param sq Vector of superquadric point clouds
     * @param sq_poses Vector of superquadric rotation matrices
     * @param target_frame Frame ID for output poses
     * @param orientation Grasp orientation: "front" or "top" (default: "front")
     * @param grasp_width Gripper width for filtering (default: 0.5)
     * @param debug Enable debug output (default: false)
     */
    Grasps(const std::vector<std::shared_ptr<open3d::geometry::PointCloud>>& sq,
           const std::vector<Eigen::Matrix3d>& sq_poses,
           const std::string& target_frame,
           const std::string& orientation = "front",
           double grasp_width = 0.5,
           bool debug = false);

    /**
     * @brief Get all potential grasp points
     * @return Point cloud of grasp candidates
     */
    std::shared_ptr<open3d::geometry::PointCloud> getGraspPoints() const { 
        return grasp_points_; 
    }

    /**
     * @brief Get the selected grasp pose
     * @return PoseStamped message with best grasp
     */
    geometry_msgs::PoseStamped getSelectedGrasps() const { 
        return selected_grasps_; 
    }

private:
    /**
     * @brief Filter and select best superquadric for grasping
     * @param sq_list Vector of superquadric point clouds
     * @param sq_poses Vector of rotation matrices
     * @return Pair of (grasp point, rotation matrix)
     */
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> graspPointFiltering(
        const std::vector<std::shared_ptr<open3d::geometry::PointCloud>>& sq_list,
        const std::vector<Eigen::Matrix3d>& sq_poses);

    /**
     * @brief Generate PoseStamped from grasp point and orientation
     * @param grasp_point 3D position of grasp
     * @param frame_id Target frame
     * @param sq_pose Rotation matrix for orientation
     * @return Generated PoseStamped message
     */
    geometry_msgs::PoseStamped generatePose(const Eigen::Vector3d& grasp_point,
                                            const std::string& frame_id,
                                            const Eigen::Matrix3d& sq_pose);

    /**
     * @brief Convert rotation matrix to quaternion
     * @param R 3x3 rotation matrix
     * @return Eigen quaternion (x, y, z, w)
     */
    Eigen::Quaterniond rotationMatrixToQuaternion(const Eigen::Matrix3d& R);

    // Member variables
    std::string target_frame_;
    std::string orientation_;
    double grasp_width_;
    bool debug_;
    
    std::shared_ptr<open3d::geometry::PointCloud> grasp_points_;
    Eigen::Vector3d primary_point_;
    Eigen::Matrix3d sq_pose_;
    geometry_msgs::PoseStamped selected_grasps_;
};

} // namespace super_grasp

#endif // SUPER_GRASP_GRASPS_H