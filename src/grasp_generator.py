#!/usr/bin/env python3

import rospy
import numpy as np
import message_filters
import time
import psutil
import os

from superquadric import Superquadric
from pointCloudData import PointCloudData
from grasps import Grasps

from geometry_msgs.msg import PoseStamped, Pose
from visualization_msgs.msg import Marker
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from unsw_vision_msgs.msg import DetectionList
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
from tf.transformations import quaternion_from_euler
import tf2_ros
import tf2_geometry_msgs  # For PoseStamped

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

class GraspGenerator():
    def __init__(self):
        rospy.init_node("grasp_generator_node")

        self.latest_rgb = None
        self.latest_depth = None
        self.latest_camera_info = None
        self.latest_detections = None

        #initalize subscribers
        self.rgb_sub = rospy.Subscriber('/hsrb/head_rgbd_sensor/rgb/image_raw', Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber('/hsrb/head_rgbd_sensor/depth_registered/image_raw', Image, self.depth_callback)
        self.camera_info_sub = rospy.Subscriber('/hsrb/head_rgbd_sensor/depth_registered/camera_info', CameraInfo, callback=self.camera_info_callback)
        self.detection_sub = rospy.Subscriber('/unsw_vision/detections/objects/positions', DetectionList, callback=self.main_callback)
        
        #initialise publishers
        self.grasp_pose_pub = rospy.Publisher('/grasp_pose', PoseStamped, queue_size=10)
        self.point_cloud_pub = rospy.Publisher('/superquadric', PointCloud2, queue_size=10)

        # Publishers in map frame
        self.grasp_pose_pub_map = rospy.Publisher('/grasp_pose_map', PoseStamped, queue_size=10)
        self.point_cloud_pub_map = rospy.Publisher('/superquadric_map', PointCloud2, queue_size=10)
        # self.marker_pub_1 = rospy.Publisher('/grasp_point_1', Marker, queue_size=10)
        # self.marker_pub_2 = rospy.Publisher('/grasp_point_2', Marker, queue_size=10)
                # TF buffer & listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Initialize process for CPU monitoring
        self.process = psutil.Process(os.getpid())

        rospy.loginfo('GraspGenerator node initialized')

    # Callback functions to store latest camera messages
    def rgb_callback(self, msg):
        self.latest_rgb = msg
        
    def depth_callback(self, msg):
        self.latest_depth = msg
        
    def camera_info_callback(self, msg):
        self.latest_camera_info = msg

    
    #main called when vision detects an obejct
    def main_callback(self, detection_msg):
        try:
            #loop through all detected objects
            if detection_msg and detection_msg.objects:

                for obj in detection_msg.objects:
                    #specify target obejct class (for test purposes)
                    if obj.object_class not in ['cup']:
                        continue

                    # Checkpoint 0: Start time and CPU usage tracking
                    start_time = time.time()
                    start_cpu, start_memory = self.get_performance_metrics()
                    rospy.loginfo(f"Starting processing for object ID: {obj.tracking_id}")

                    pointCloudObject = PointCloudData(
                        object_ID=obj.tracking_id,
                        raw_rgb=self.latest_rgb,
                        raw_depth=self.latest_depth,
                        mask=obj.mask,
                        camera_info=self.latest_camera_info
                    )

                    # Checkpoint 1: PCD processing complete
                    self.print_checkpoint("Checkpoint 1: PCD Processing Complete", start_time, start_cpu, start_memory)
                    checkpoint1_time = time.time()
                    checkpoint1_cpu, checkpoint1_memory = self.get_performance_metrics()

                    superquadric = Superquadric(
                        object_ID=obj.tracking_id,
                        class_name=obj.object_class,
                        pcd=pointCloudObject
                    )

                    # Checkpoint 2: Superquadric processing complete
                    self.print_checkpoint("Checkpoint 2: Superquadric Processing Complete", checkpoint1_time, checkpoint1_cpu, checkpoint1_memory)

                    graspPose = Grasps(
                        sq=superquadric,
                        orientation='front'
                    ).getSelectedGrasps()
                
                        
                    # Print total processing time
                    total_time = time.time() - start_time
                    final_cpu, final_memory = self.get_performance_metrics()
                    rospy.loginfo(f"=== TOTAL PROCESSING TIME ===")
                    rospy.loginfo(f"Total elapsed time: {total_time:.3f} seconds")
                    rospy.loginfo(f"Final CPU usage: {final_cpu:.1f}%")
                    rospy.loginfo(f"Total memory change: {final_memory - start_memory:.1f} MB")
                    rospy.loginfo("=" * 30)

                    #publish grasp pose and obejct model
                    if superquadric and graspPose:
                        
                        self.publish_in_both_frames(graspPose)
                        # self.grasp_pose_pub.publish(graspPose)
                        self.point_cloud_pub.publish(self.convert_o3d_to_ros_cloud(superquadric.getSuperquadricAsPCD()))
                    else:
                        rospy.loginfo("No valid superquadric model or grasp pose to publish.")
                        
            else:
                rospy.loginfo("No detected objects.")
        except Exception as e:
            rospy.logerr(f"Error in synchronized callback: {e}")
    
    def publish_in_both_frames(self, grasp_pose):
        # Publish in camera frame
        self.grasp_pose_pub.publish(grasp_pose)

        try:
            # Transform pose
            transform = self.tf_buffer.lookup_transform(
                "map",  # target
                grasp_pose.header.frame_id,  # source
                rospy.Time(0),
                rospy.Duration(1.0)
            )
            grasp_pose_map = tf2_geometry_msgs.do_transform_pose(grasp_pose, transform)

            # Transform cloud
            # cloud_map = tf2_sensor_msgs.do_transform_cloud(cloud_msg, transform)

            # Publish in map frame
            self.grasp_pose_pub_map.publish(grasp_pose_map)
            # self.point_cloud_pub_map.publish(cloud_map)

        except Exception as e:
            rospy.logwarn(f"TF transform failed: {e}")

    #convert superquadric from open3d point cloud to roscloud: debugging purposes
    def convert_o3d_to_ros_cloud(self, cloud_o3d, frame_id="head_rgbd_sensor_rgb_frame"):
        try:
            points = np.asarray(cloud_o3d.points)
            colors = np.asarray(cloud_o3d.colors)

            if points.shape[0] == 0:
                return None

            rgb_packed = (colors * 255).astype(np.uint8)
            rgb_packed = rgb_packed[:, 0] << 16 | rgb_packed[:, 1] << 8 | rgb_packed[:, 2] 

            cloud_data = [[x, y, z] for (x, y, z) in points]

            fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1)
            ]

            cloud_msg = pc2.create_cloud(rospy.Header(frame_id=frame_id, stamp=rospy.Time.now()), fields, cloud_data)
            return cloud_msg
        except Exception as e:
            rospy.logerr(f"PCD to ROS conversion Error: {e}")
            return None

    def get_performance_metrics(self):
        """Get current CPU usage and memory usage"""
        cpu_percent = self.process.cpu_percent()
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
        return cpu_percent, memory_mb

    def print_checkpoint(self, checkpoint_name, start_time, start_cpu, start_memory):
        """Print performance metrics for a checkpoint"""
        current_time = time.time()
        current_cpu, current_memory = self.get_performance_metrics()
        
        elapsed_time = current_time - start_time
        cpu_usage = current_cpu  # Current CPU usage percentage
        memory_usage = current_memory
        
        rospy.loginfo(f"=== {checkpoint_name} ===")
        rospy.loginfo(f"Elapsed time: {elapsed_time:.3f} seconds")
        rospy.loginfo(f"CPU usage: {cpu_usage:.1f}%")
        rospy.loginfo(f"Memory usage: {memory_usage:.1f} MB")
        rospy.loginfo(f"Memory change: {memory_usage - start_memory:.1f} MB")
        rospy.loginfo("=" * 30)

if __name__ == "__main__":
    node = GraspGenerator()
    rospy.spin()