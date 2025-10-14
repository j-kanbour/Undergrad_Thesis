#include "super_grasp/point_cloud_data.h"
#include <chrono>

namespace super_grasp {

PointCloudData::PointCloudData(const sensor_msgs::ImageConstPtr& raw_rgb,
                               const sensor_msgs::ImageConstPtr& raw_depth,
                               const std::vector<geometry_msgs::Point>& mask,
                               const sensor_msgs::CameraInfoConstPtr& camera_info,
                               int nearest_neighbor,
                               double distance_thresh,
                               int segment_threshold,
                               bool debug)
    : mask_(mask),
      camera_info_(camera_info),
      nearest_neighbor_(nearest_neighbor),
      distance_thresh_(distance_thresh),
      segment_threshold_(segment_threshold),
      debug_(debug) {

    auto start_time = std::chrono::high_resolution_clock::now();

    // Convert ROS images to OpenCV
    try {
        cv_bridge::CvImageConstPtr cv_rgb = cv_bridge::toCvShare(raw_rgb, "bgr8");
        cv_bridge::CvImageConstPtr cv_depth = cv_bridge::toCvShare(raw_depth);
        raw_rgb_ = cv_rgb->image.clone();
        raw_depth_ = cv_depth->image.clone();
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    // Convert to point cloud
    pcd_ = convertToPCD();

    if (debug_) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        ROS_INFO("pointCloudData: Point cloud generated within %.3fs", duration.count() / 1000.0);
    }
}

std::shared_ptr<open3d::geometry::PointCloud> PointCloudData::removeOutliers(
    std::shared_ptr<open3d::geometry::PointCloud> pcd) {
    
    try {
        if (pcd->IsEmpty()) {
            ROS_WARN("pointCloudData [removeOutliers]: Provided point cloud is empty.");
            return pcd;
        }

        // Statistical outlier removal
        auto [cleaned_pcd, ind] = pcd->RemoveStatisticalOutliers(nearest_neighbor_, 0.25);
        
        return cleaned_pcd;
    } catch (const std::exception& e) {
        ROS_ERROR("pointCloudData [removeOutliers] Error: %s", e.what());
        return pcd;
    }
}

std::shared_ptr<open3d::geometry::PointCloud> PointCloudData::convertToPCD() {
    try {
        // Create mask image
        cv::Mat mask = cv::Mat::zeros(camera_info_->height, camera_info_->width, CV_8UC1);
        
        std::vector<cv::Point> points;
        for (const auto& pt : mask_) {
            points.push_back(cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)));
        }
        
        cv::fillPoly(mask, points, cv::Scalar(255));

        // Apply mask to RGB and depth
        cv::Mat rgb_masked = cv::Mat::zeros(raw_rgb_.size(), raw_rgb_.type());
        cv::Mat depth_masked = cv::Mat::zeros(raw_depth_.size(), raw_depth_.type());
        
        raw_rgb_.copyTo(rgb_masked, mask);
        raw_depth_.copyTo(depth_masked, mask);

        // Ensure depth is in correct format (mm as uint16 or meters as float)
        cv::Mat depth_float;
        if (depth_masked.type() == CV_16UC1) {
            depth_masked.convertTo(depth_float, CV_32F, 1.0 / 1000.0); // Convert mm to meters
        } else if (depth_masked.type() == CV_32FC1) {
            depth_float = depth_masked;
        } else {
            depth_masked.convertTo(depth_float, CV_32F);
        }

        // Create Open3D images
        auto color_open3d = std::make_shared<open3d::geometry::Image>();
        auto depth_open3d = std::make_shared<open3d::geometry::Image>();
        
        color_open3d->Prepare(raw_rgb_.cols, raw_rgb_.rows, 3, 1);
        depth_open3d->Prepare(depth_float.cols, depth_float.rows, 1, 4);

        // Copy data
        memcpy(color_open3d->data_.data(), rgb_masked.data, 
               rgb_masked.total() * rgb_masked.elemSize());
        memcpy(depth_open3d->data_.data(), depth_float.data,
               depth_float.total() * depth_float.elemSize());

        // Create RGBD image
        auto rgbd = open3d::geometry::RGBDImage::CreateFromColorAndDepth(
            *color_open3d, *depth_open3d, 1000.0, 10.0, false);

        // Extract camera intrinsics
        double fx = camera_info_->K[0];
        double fy = camera_info_->K[4];
        double cx = camera_info_->K[2];
        double cy = camera_info_->K[5];
        int width = camera_info_->width;
        int height = camera_info_->height;

        open3d::camera::PinholeCameraIntrinsic intrinsic(width, height, fx, fy, cx, cy);

        // Generate point cloud
        auto point_cloud = open3d::geometry::PointCloud::CreateFromRGBDImage(*rgbd, intrinsic);

        if (debug_) {
            ROS_INFO("pointCloudData: pointcloud size %zu", point_cloud->points_.size());
        }

        // Remove outliers
        point_cloud = removeOutliers(point_cloud);

        return point_cloud;

    } catch (const std::exception& e) {
        ROS_ERROR("pointCloudData [convertToPCD] Error: %s", e.what());
        return std::make_shared<open3d::geometry::PointCloud>();
    }
}

std::vector<std::shared_ptr<open3d::geometry::PointCloud>> 
PointCloudData::defineSegments(std::shared_ptr<open3d::geometry::PointCloud> pcd) {
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();

        size_t remaining_points_threshold = pcd->points_.size() / segment_threshold_;
        std::vector<std::shared_ptr<open3d::geometry::PointCloud>> cloud_segments;
        
        auto remaining = std::make_shared<open3d::geometry::PointCloud>(*pcd);

        while (remaining->points_.size() > remaining_points_threshold) {
            // Segment plane using RANSAC
            auto [plane_model, inliers] = remaining->SegmentPlane(
                distance_thresh_, 3, 1000, 0.999);

            if (inliers.size() == 0) {
                break;
            }

            auto inlier_cloud = remaining->SelectByIndex(inliers);
            cloud_segments.push_back(inlier_cloud);

            remaining = remaining->SelectByIndex(inliers, true); // Invert selection
        }

        if (debug_) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            ROS_INFO("pointCloudData: Segmented into %zu planes. Generated within %.3fs",
                     cloud_segments.size(), duration.count() / 1000.0);
        }

        return cloud_segments;

    } catch (const std::exception& e) {
        ROS_ERROR("pointCloudData [defineSegments] Error: %s", e.what());
        return std::vector<std::shared_ptr<open3d::geometry::PointCloud>>();
    }
}

std::vector<std::shared_ptr<open3d::geometry::PointCloud>> 
PointCloudData::getCloudSegments() {
    return defineSegments(pcd_);
}

} // namespace super_grasp