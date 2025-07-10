#!/usr/bin/env python3.8

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
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

class GraspGenerator():
    def __init__(self):
        rospy.init_node("grasp_generator_node")

        self.rgb_sub = message_filters.Subscriber('/hsrb/head_rgbd_sensor/rgb/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('/hsrb/head_rgbd_sensor/depth_registered/image_raw', Image)
        self.camera_info_sub = rospy.Subscriber('/hsrb/head_rgbd_sensor/depth_registered/camera_info', CameraInfo, callback=self.camera_info_callback)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1
        )
        self.sync.registerCallback(self.main_callback)

        self.detection_msg = []
        #unsw_vision/detections/objects/positions
        self.detection_sub = rospy.Subscriber('/unsw_vision/detections/objects/positions', DetectionList, callback=self.detections_callback)
        self.last_detection_time = rospy.Time.now()
        self.detection_timer = rospy.Timer(rospy.Duration(2.0), self.check_detection_timeout)

        self.point_cloud_pub = rospy.Publisher('/superquadric_pointcloud', PointCloud2, queue_size=10)
        self.grasp_pose_pub = rospy.Publisher('/grasp_pose', PoseStamped, queue_size=10)

        self.marker_pub_1 = rospy.Publisher('/grasp_point_1', Marker, queue_size=10)
        self.marker_pub_2 = rospy.Publisher('/grasp_point_2', Marker, queue_size=10)

        self.latest_camera_info = None

        # Initialize process for CPU monitoring
        self.process = psutil.Process(os.getpid())

        rospy.loginfo('GraspGenerator node initialized')

    def convert_pose_dict_to_ros_pose(self, pose_dict):
        """
        Convert pose dictionary to ROS Pose message
        pose_dict format: {"origin": [x,y,z], "vector": [x,y,z]}
        """
        pose = Pose()
        
        # Set position from origin
        pose.position.x = pose_dict["origin"][0]
        pose.position.y = pose_dict["origin"][1] 
        pose.position.z = pose_dict["origin"][2]
        
        # Convert vector to quaternion orientation
        # Assuming the vector represents the direction the gripper should approach
        vector = np.array(pose_dict["vector"])
        vector_norm = np.linalg.norm(vector)
        
        if vector_norm > 0:
            vector = vector / vector_norm  # Normalize
            
            # Calculate roll, pitch, yaw from the vector
            # This is a simple approach - you might need to adjust based on your coordinate system
            yaw = np.arctan2(vector[1], vector[0])
            pitch = np.arcsin(-vector[2])
            roll = 0.0  # Assuming no roll for simplicity
            
            # Convert to quaternion
            quaternion = quaternion_from_euler(roll, pitch, yaw)
            pose.orientation.x = quaternion[0]
            pose.orientation.y = quaternion[1]
            pose.orientation.z = quaternion[2]
            pose.orientation.w = quaternion[3]
        else:
            # Default orientation if vector is zero
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            pose.orientation.w = 1.0
        
        return pose

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

    def camera_info_callback(self, msg):
        self.latest_camera_info = msg

    def detections_callback(self, detection_msg):
        self.last_detection_time = rospy.Time.now()
        self.detection_msg = detection_msg

    def check_detection_timeout(self, _):
        if (rospy.Time.now() - self.last_detection_time).to_sec() > 2.0:
            self.detection_msg = []

    def main_callback(self, rgb_msg, depth_msg):
        try:
            if self.detection_msg and self.detection_msg.objects:

                all_models = []
                for obj in self.detection_msg.objects:
                    if obj.object_class not in []:

                        # Checkpoint 0: Start time and CPU usage tracking
                        start_time = time.time()
                        start_cpu, start_memory = self.get_performance_metrics()
                        rospy.loginfo(f"Starting processing for object ID: {obj.tracking_id}")

                        pcd = PointCloudData(
                            object_ID=obj.tracking_id,
                            raw_data_1=rgb_msg,
                            bbox=obj.bbox,
                            raw_depth=depth_msg,
                            raw_mask=None,
                            camera_info=self.latest_camera_info
                        )

                        # Checkpoint 1: PCD processing complete
                        self.print_checkpoint("Checkpoint 1: PCD Processing Complete", 
                                            start_time, start_cpu, start_memory)
                        checkpoint1_time = time.time()
                        checkpoint1_cpu, checkpoint1_memory = self.get_performance_metrics()

                        superquadric = Superquadric(
                            object_ID=obj.tracking_id,
                            class_name=obj.object_class,
                            pcd=pcd
                        )

                        # Checkpoint 2: Superquadric processing complete
                        self.print_checkpoint("Checkpoint 2: Superquadric Processing Complete", 
                                            checkpoint1_time, checkpoint1_cpu, checkpoint1_memory)


                        if superquadric:
                            all_models.append(superquadric.getSuperquadricAsPCD())
                    
                            try:
                                grasp_obj = Grasps(superquadric, 'front', False)
                                grasp_pose = grasp_obj.selectGrasps()

                                if grasp_pose:
                                    pose_stamped = PoseStamped()
                                    pose_stamped.header = rgb_msg.header
                                    
                                    # Convert the pose dictionary to ROS Pose message
                                    pose_stamped.pose = self.convert_pose_dict_to_ros_pose(grasp_pose["pose"])
                                    
                                    # Publish the pose
                                    self.grasp_pose_pub.publish(pose_stamped)

                                    point_1 = grasp_pose["point_1"]
                                    point_2 = grasp_pose["point_2"]

                                    marker_1 = self.publish_grasp_point_marker(point_1, rgb_msg.header.frame_id, marker_id=0, color=(1.0, 0.0, 0.0))
                                    marker_2 = self.publish_grasp_point_marker(point_2, rgb_msg.header.frame_id, marker_id=1, color=(0.0, 0.0, 1.0))

                                    self.marker_pub_1.publish(marker_1)
                                    self.marker_pub_2.publish(marker_2)

                                    rospy.loginfo("Published grasp pose and grasp points.")

                                    self.print_checkpoint("Checkpoint 3: Grasp Processing Complete", 
                                            checkpoint1_time, checkpoint1_cpu, checkpoint1_memory)
                                else:
                                    rospy.logwarn("No valid grasp pose found.")
                            except Exception as e:
                                rospy.logerr(f'grasps Error: {e}')
                                import traceback
                                traceback.print_exc()

                            
                        # Print total processing time
                        total_time = time.time() - start_time
                        final_cpu, final_memory = self.get_performance_metrics()
                        rospy.loginfo(f"=== TOTAL PROCESSING TIME ===")
                        rospy.loginfo(f"Total elapsed time: {total_time:.3f} seconds")
                        rospy.loginfo(f"Final CPU usage: {final_cpu:.1f}%")
                        rospy.loginfo(f"Total memory change: {final_memory - start_memory:.1f} MB")
                        rospy.loginfo("=" * 30)

                if all_models:
                    self.publish_combined_point_cloud(all_models, rgb_msg.header)
                else:
                    rospy.loginfo("No valid superquadric models to publish.")
            else:
                rospy.loginfo("No detected objects.")
        except Exception as e:
            rospy.logerr(f"Error in synchronized callback: {e}")

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

    def publish_combined_point_cloud(self, models, header):
        for model in models:
            cloud_msg = self.convert_o3d_to_ros_cloud(model, frame_id=header.frame_id)
            if cloud_msg:
                self.point_cloud_pub.publish(cloud_msg)
                rospy.loginfo("Published point cloud for object")

    def publish_grasp_point_marker(self, point, frame_id, marker_id, color):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "grasp_points"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = point[0]
        marker.pose.position.y = point[1]
        marker.pose.position.z = point[2]
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.02
        marker.scale.y = 0.02
        marker.scale.z = 0.02

        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0

        marker.lifetime = rospy.Duration(5.0)
        return marker

if __name__ == "__main__":
    node = GraspGenerator()
    rospy.spin()