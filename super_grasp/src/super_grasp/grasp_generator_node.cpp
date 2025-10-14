#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>
#include <unsw_vision_msgs/DetectionList.h>

#include "super_grasp/point_cloud_data.h"
#include "super_grasp/superquadric.h"
#include "super_grasp/grasps.h"

#include <chrono>
#include <memory>

class GraspGeneratorNode {
public:
    GraspGeneratorNode() : tf_buffer_(), tf_listener_(tf_buffer_) {
        ros::NodeHandle nh;
        ros::NodeHandle private_nh("~");

        // Load parameters
        private_nh.param<bool>("debug", debug_, false);
        private_nh.param<double>("grasp_width", grasp_width_, 0.236);
        private_nh.param<std::string>("rgb_topic", rgb_topic_, 
                                      "/hsrb/head_rgbd_sensor/rgb/image_raw");
        private_nh.param<std::string>("depth_topic", depth_topic_,
                                      "/hsrb/head_rgbd_sensor/depth_registered/image_raw");
        private_nh.param<std::string>("camera_info_topic", camera_info_topic_,
                                      "/hsrb/head_rgbd_sensor/depth_registered/camera_info");
        private_nh.param<std::string>("detections_topic", detections_topic_,
                                      "/unsw_vision/detections/objects/positions");
        private_nh.param<int>("target_object_id", target_object_id_, -1);
        private_nh.param<std::string>("target_object_class", target_object_class_, "false");
        private_nh.param<std::string>("orientation", orientation_, "front");

        // Initialize subscribers
        rgb_sub_ = nh.subscribe(rgb_topic_, 1, &GraspGeneratorNode::rgbCallback, this);
        depth_sub_ = nh.subscribe(depth_topic_, 1, &GraspGeneratorNode::depthCallback, this);
        camera_info_sub_ = nh.subscribe(camera_info_topic_, 1, 
                                       &GraspGeneratorNode::cameraInfoCallback, this);
        detection_sub_ = nh.subscribe(detections_topic_, 1,
                                     &GraspGeneratorNode::mainCallback, this);

        // Initialize publishers
        grasp_pose_pub_base_ = nh.advertise<geometry_msgs::PoseStamped>("grasp_pose_base", 10);
        grasp_pose_pub_hand_ = nh.advertise<geometry_msgs::PoseStamped>("grasp_pose_hand", 10);

        if (debug_) {
            grasp_pose_pub_camera_ = nh.advertise<geometry_msgs::PoseStamped>(
                "grasp_pose_camera", 10);
            superquadric_pub_ = nh.advertise<sensor_msgs::PointCloud2>("superquadric", 10);
            point_cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>(
                "extracted_point_cloud", 10);
            superquadric_bbox_pub_ = nh.advertise<visualization_msgs::Marker>(
                "superquadric_bbox", 10);
        }

        ROS_INFO("GraspGenerator node initialized");
    }

    void rgbCallback(const sensor_msgs::ImageConstPtr& msg) {
        latest_rgb_ = msg;
    }

    void depthCallback(const sensor_msgs::ImageConstPtr& msg) {
        latest_depth_ = msg;
    }

    void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg) {
        camera_info_ = msg;
    }

    void mainCallback(const unsw_vision_msgs::DetectionListConstPtr& detection_msg) {
        try {
            if (!run_ || !detection_msg || detection_msg->objects.empty()) {
                if (!detection_msg || detection_msg->objects.empty()) {
                    ROS_INFO_THROTTLE(5.0, "No detected objects.");
                }
                return;
            }

            if (debug_) {
                ROS_INFO("START");
            }

            // Process each detected object
            for (const auto& obj : detection_msg->objects) {
                // Filter by object ID or class if specified
                if ((target_object_id_ != -1 && obj.tracking_id != target_object_id_) &&
                    (target_object_class_ != "false" && obj.object_class != target_object_class_)) {
                    continue;
                }

                auto init_time = std::chrono::high_resolution_clock::now();

                // Generate point cloud from RGB, depth and mask
                super_grasp::PointCloudData pcd(
                    latest_rgb_,
                    latest_depth_,
                    obj.mask.points,
                    camera_info_,
                    500,   // nearest_neighbor
                    0.005, // distance_thresh
                    10,    // segment_threshold
                    debug_
                );

                auto pcd_time = std::chrono::high_resolution_clock::now();

                // Extract segments from point cloud
                auto cloud_segments = pcd.getCloudSegments();

                auto cloud_segment_time = std::chrono::high_resolution_clock::now();

                std::vector<std::shared_ptr<open3d::geometry::PointCloud>> superquadrics;
                std::vector<Eigen::Matrix3d> sq_poses;

                // Build all SQs for debugging
                std::shared_ptr<open3d::geometry::PointCloud> all_sqs;
                std::shared_ptr<open3d::geometry::PointCloud> all_clouds;
                
                if (debug_) {
                    all_sqs = std::make_shared<open3d::geometry::PointCloud>();
                    all_clouds = std::make_shared<open3d::geometry::PointCloud>();
                }

                // Generate superquadric fits for each segment
                for (const auto& segment : cloud_segments) {
                    super_grasp::Superquadric sq(segment, 30, debug_);

                    auto sq_mesh = sq.getSuperquadricMesh();
                    auto sq_pose = sq.getSQPose();

                    superquadrics.push_back(sq_mesh);
                    sq_poses.push_back(sq_pose);

                    if (debug_) {
                        *all_sqs += *sq_mesh;
                        *all_clouds += *segment;
                    }
                }

                auto sq_time = std::chrono::high_resolution_clock::now();

                // Generate and select grasp from superquadrics
                super_grasp::Grasps grasps(
                    superquadrics,
                    sq_poses,
                    camera_info_->header.frame_id,
                    orientation_,
                    grasp_width_,
                    debug_
                );

                geometry_msgs::PoseStamped grasp_pose = grasps.getSelectedGrasps();

                auto grasp_time = std::chrono::high_resolution_clock::now();

                // Publish grasp in base frame
                geometry_msgs::PoseStamped base_frame_pose = cameraToBaseTransform(grasp_pose);
                if (base_frame_pose.header.frame_id != "") {
                    grasp_pose_pub_base_.publish(base_frame_pose);
                }

                // Publish grasp in hand frame
                geometry_msgs::PoseStamped hand_frame_pose = cameraToHandTransform(grasp_pose);
                if (hand_frame_pose.header.frame_id != "") {
                    grasp_pose_pub_hand_.publish(hand_frame_pose);
                }

                if (debug_) {
                    // Publish superquadric
                    if (all_sqs && !all_sqs->IsEmpty()) {
                        sensor_msgs::PointCloud2 sq_msg = convertO3DToROSCloud(
                            all_sqs, camera_info_->header.frame_id);
                        superquadric_pub_.publish(sq_msg);
                    }

                    // Publish segment point clouds
                    if (all_clouds && !all_clouds->IsEmpty()) {
                        sensor_msgs::PointCloud2 cloud_msg = convertO3DToROSCloud(
                            all_clouds, camera_info_->header.frame_id);
                        point_cloud_pub_.publish(cloud_msg);
                    }

                    // Publish grasp in camera frame
                    grasp_pose_pub_camera_.publish(grasp_pose);

                    // Print timing information
                    auto total_time = grasp_time;
                    ROS_INFO("\n\n\nTiming breakdown:");
                    ROS_INFO("PCD time: %.3fs",
                             std::chrono::duration<double>(pcd_time - init_time).count());
                    ROS_INFO("Segmentation time: %.3fs",
                             std::chrono::duration<double>(cloud_segment_time - pcd_time).count());
                    ROS_INFO("SQ time: %.3fs",
                             std::chrono::duration<double>(sq_time - cloud_segment_time).count());
                    ROS_INFO("Grasp time: %.3fs",
                             std::chrono::duration<double>(grasp_time - sq_time).count());
                    ROS_INFO("Total time: %.3fs\n\n\n",
                             std::chrono::duration<double>(grasp_time - init_time).count());

                    ROS_INFO("END");
                }
            }

        } catch (const std::exception& e) {
            ROS_ERROR("Error in main callback: %s", e.what());
        }
    }

private:
    geometry_msgs::PoseStamped cameraToBaseTransform(
        const geometry_msgs::PoseStamped& grasp_pose) {
        
        std::string target_frame = "base_footprint";
        
        try {
            // Lookup transform
            geometry_msgs::TransformStamped transform = tf_buffer_.lookupTransform(
                target_frame, grasp_pose.header.frame_id, ros::Time(0), ros::Duration(0.2));

            // Transform the pose
            geometry_msgs::PoseStamped pose;
            tf2::doTransform(grasp_pose, pose, transform);

            // Calculate yaw to point towards the object
            double dx = pose.pose.position.x;
            double dy = pose.pose.position.y;
            double yaw = std::atan2(dy, dx);

            // Create horizontal orientation pointing at object
            Eigen::Quaterniond q(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
            Eigen::Matrix3d R_horiz = q.toRotationMatrix();

            // Move 50cm backwards along forward (x) axis
            Eigen::Vector3d forward_vector = R_horiz.col(0);
            Eigen::Vector3d delta_pos = -0.5 * forward_vector;
            Eigen::Vector3d current_pos(pose.pose.position.x, 
                                       pose.pose.position.y, 
                                       pose.pose.position.z);
            Eigen::Vector3d new_pos = current_pos + delta_pos;

            // Build output pose
            geometry_msgs::PoseStamped base_pose;
            base_pose.header.frame_id = "base_footprint";
            base_pose.header.stamp = ros::Time::now();
            base_pose.pose.position.x = new_pos(0);
            base_pose.pose.position.y = new_pos(1);
            base_pose.pose.position.z = 0.0;
            base_pose.pose.orientation.x = q.x();
            base_pose.pose.orientation.y = q.y();
            base_pose.pose.orientation.z = q.z();
            base_pose.pose.orientation.w = q.w();

            return base_pose;

        } catch (const tf2::TransformException& e) {
            ROS_WARN("camera to base transform: %s", e.what());
            return geometry_msgs::PoseStamped();
        }
    }

    geometry_msgs::PoseStamped cameraToHandTransform(
        const geometry_msgs::PoseStamped& grasp_pose) {
        
        std::string target_frame = "hand_palm_link";

        if (grasp_pose.header.frame_id.empty()) {
            ROS_ERROR("camera_to_hand_transform received empty grasp_pose");
            return geometry_msgs::PoseStamped();
        }

        try {
            // Use Time(0) to get latest available transform
            geometry_msgs::TransformStamped transform = tf_buffer_.lookupTransform(
                target_frame, grasp_pose.header.frame_id, ros::Time(0), ros::Duration(2.0));

            geometry_msgs::PoseStamped hand_pose;
            tf2::doTransform(grasp_pose, hand_pose, transform);

            // Update timestamp to match transform
            hand_pose.header.stamp = transform.header.stamp;

            return hand_pose;

        } catch (const tf2::TransformException& e) {
            ROS_WARN("Camera to hand transform: %s", e.what());
            return geometry_msgs::PoseStamped();
        }
    }

    sensor_msgs::PointCloud2 convertO3DToROSCloud(
        std::shared_ptr<open3d::geometry::PointCloud> cloud,
        const std::string& frame_id) {
        
        sensor_msgs::PointCloud2 cloud_msg;
        cloud_msg.header.frame_id = frame_id;
        cloud_msg.header.stamp = ros::Time::now();
        
        cloud_msg.height = 1;
        cloud_msg.width = cloud->points_.size();
        cloud_msg.is_dense = false;
        cloud_msg.is_bigendian = false;

        sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
        modifier.setPointCloud2FieldsByString(1, "xyz");

        sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");

        for (size_t i = 0; i < cloud->points_.size(); ++i, ++iter_x, ++iter_y, ++iter_z) {
            *iter_x = cloud->points_[i](0);
            *iter_y = cloud->points_[i](1);
            *iter_z = cloud->points_[i](2);
        }

        return cloud_msg;
    }

    // Member variables
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    sensor_msgs::ImageConstPtr latest_rgb_;
    sensor_msgs::ImageConstPtr latest_depth_;
    sensor_msgs::CameraInfoConstPtr camera_info_;
    bool run_ = true;

    // Parameters
    bool debug_;
    double grasp_width_;
    std::string rgb_topic_;
    std::string depth_topic_;
    std::string camera_info_topic_;
    std::string detections_topic_;
    int target_object_id_;
    std::string target_object_class_;
    std::string orientation_;

    // Subscribers
    ros::Subscriber rgb_sub_;
    ros::Subscriber depth_sub_;
    ros::Subscriber camera_info_sub_;
    ros::Subscriber detection_sub_;

    // Publishers
    ros::Publisher grasp_pose_pub_base_;
    ros::Publisher grasp_pose_pub_hand_;
    ros::Publisher grasp_pose_pub_camera_;
    ros::Publisher superquadric_pub_;
    ros::Publisher point_cloud_pub_;
    ros::Publisher superquadric_bbox_pub_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "grasp_generator_node");
    
    GraspGeneratorNode node;
    
    ros::spin();
    
    return 0;
}