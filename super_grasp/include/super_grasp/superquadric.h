#ifndef SUPER_GRASP_SUPERQUADRIC_H
#define SUPER_GRASP_SUPERQUADRIC_H

#include <open3d/Open3D.h>
#include <Eigen/Dense>
#include <memory>

namespace super_grasp {

/**
 * @brief Fits superquadric models to point cloud segments
 * 
 * Estimates superquadric shape parameters (e1, e2) based on point cloud kurtosis
 * and size parameters (a1, a2, a3) from oriented bounding box.
 */
class Superquadric {
public:
    /**
     * @brief Constructor
     * @param pcd Input point cloud segment
     * @param downsample Downsampling percentage for superquadric points (default: 30)
     * @param debug Enable debug output (default: false)
     */
    Superquadric(std::shared_ptr<open3d::geometry::PointCloud> pcd,
                 int downsample = 30,
                 bool debug = false);

    /**
     * @brief Get the fitted superquadric as a point cloud
     * @return Superquadric point cloud
     */
    std::shared_ptr<open3d::geometry::PointCloud> getSuperquadricMesh() const { 
        return superquadric_; 
    }

    /**
     * @brief Get the pose (rotation matrix) of the superquadric
     * @return 3x3 rotation matrix
     */
    Eigen::Matrix3d getSQPose() const { return pose_; }

private:
    /**
     * @brief Estimate shape parameters e1 and e2 from kurtosis
     * @param pcd Input point cloud
     * @return Pair of (e1, e2) values
     */
    std::pair<double, double> estimateE(std::shared_ptr<open3d::geometry::PointCloud> pcd);

    /**
     * @brief Create superquadric point cloud fitted to input
     * @param pcd Input point cloud
     * @param e1 Shape parameter along z-axis
     * @param e2 Shape parameter in xy-plane
     * @return Pair of (superquadric point cloud, rotation matrix)
     */
    std::pair<std::shared_ptr<open3d::geometry::PointCloud>, Eigen::Matrix3d> 
    createSuperquadric(std::shared_ptr<open3d::geometry::PointCloud> pcd, 
                       double e1, double e2);

    /**
     * @brief Calculate kurtosis along each axis
     * @param points Centered point cloud data
     * @return Vector3d of kurtosis values for x, y, z
     */
    Eigen::Vector3d calculateKurtosis(const Eigen::MatrixXd& points);

    /**
     * @brief Sign function that handles zero
     * @param x Input value
     * @return Sign of x
     */
    inline double sgn(double x) const { return (x >= 0.0) ? 1.0 : -1.0; }

    // Member variables
    double e1_;
    double e2_;
    int downsample_;
    bool debug_;
    std::shared_ptr<open3d::geometry::PointCloud> superquadric_;
    Eigen::Matrix3d pose_;
};

} // namespace super_grasp

#endif // SUPER_GRASP_SUPERQUADRIC_H