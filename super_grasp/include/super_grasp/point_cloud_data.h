#ifndef SUPER_GRASP_POINT_CLOUD_DATA_H
#define SUPER_GRASP_POINT_CLOUD_DATA_H

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <open3d/Open3D.h>
#include <vector>
#include <memory>

namespace super_grasp {

/**
 * @brief Extracts point cloud data from RGB-D images with masking and segmentation
 * 
 * This class converts raw RGB and depth images into point clouds, applies object masks,
 * removes outliers, and performs RANSAC-based plane segmentation.
 */
class PointCloudData {
public:
    /**
     * @brief Constructor
     * @param raw_rgb Raw RGB image from camera
     * @param raw_depth Raw depth image from camera
     * @param mask Polygon mask points defining the object region
     * @param camera_info Camera calibration information
     * @param nearest_neighbor Number of neighbors for outlier removal (default: 500)
     * @param distance_thresh Distance threshold for RANSAC segmentation (default: 0.005)
     * @param segment_threshold Minimum points threshold for segmentation (default: 10)
     * @param debug Enable debug output (default: false)
     */
    PointCloudData(const sensor_msgs::ImageConstPtr& raw_rgb,
                   const sensor_msgs::ImageConstPtr& raw_depth,
                   const std::vector<geometry_msgs::Point>& mask,
                   const sensor_msgs::CameraInfoConstPtr& camera_info,
                   int nearest_neighbor = 500,
                   double distance_thresh = 0.005,
                   int segment_threshold = 10,
                   bool debug = false);

    /**
     * @brief Get the generated point cloud
     * @return Shared pointer to the point cloud
     */
    std::shared_ptr<open3d::geometry::PointCloud> getPCD() const { return pcd_; }

    /**
     * @brief Segment the point cloud into multiple planes using RANSAC
     * @return Vector of point cloud segments
     */
    std::vector<std::shared_ptr<open3d::geometry::PointCloud>> getCloudSegments();

private:
    /**
     * @brief Remove statistical outliers from point cloud
     * @param pcd Input point cloud
     * @return Cleaned point cloud
     */
    std::shared_ptr<open3d::geometry::PointCloud> removeOutliers(
        std::shared_ptr<open3d::geometry::PointCloud> pcd);

    /**
     * @brief Convert masked RGB-D images to point cloud
     * @return Generated point cloud
     */
    std::shared_ptr<open3d::geometry::PointCloud> convertToPCD();

    /**
     * @brief Perform RANSAC-based plane segmentation
     * @param pcd Input point cloud
     * @return Vector of segmented point clouds
     */
    std::vector<std::shared_ptr<open3d::geometry::PointCloud>> defineSegments(
        std::shared_ptr<open3d::geometry::PointCloud> pcd);

    // Member variables
    cv::Mat raw_rgb_;
    cv::Mat raw_depth_;
    std::vector<geometry_msgs::Point> mask_;
    sensor_msgs::CameraInfoConstPtr camera_info_;
    int nearest_neighbor_;
    double distance_thresh_;
    int segment_threshold_;
    bool debug_;
    
    std::shared_ptr<open3d::geometry::PointCloud> pcd_;
};

} // namespace super_grasp

#endif // SUPER_GRASP_POINT_CLOUD_DATA_H