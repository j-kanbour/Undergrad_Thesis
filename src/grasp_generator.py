#!/usr/bin/env python3.8

import rospy
import numpy as np
import message_filters

from superquadric import Superquadric
from grasps import Grasps

from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from unsw_vision_msgs.msg import DetectionList
import sensor_msgs.point_cloud2 as pc2

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
        self.detection_sub = rospy.Subscriber('/unsw_vision/detections/objects/positions', DetectionList, callback=self.detections_callback)
        self.last_detection_time = rospy.Time.now()
        self.detection_timer = rospy.Timer(rospy.Duration(2.0), self.check_detection_timeout)

        self.point_cloud_pub = rospy.Publisher('/superquadric_pointcloud', PointCloud2, queue_size=10)
        self.grasp_pose_pub = rospy.Publisher('/grasp_pose', PoseStamped, queue_size=10)

        self.marker_pub_1 = rospy.Publisher('/grasp_point_1', Marker, queue_size=10)
        self.marker_pub_2 = rospy.Publisher('/grasp_point_2', Marker, queue_size=10)

        self.latest_camera_info = None

        rospy.loginfo('GraspGenerator node initialized')

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
                    if obj.object_class in ['cup']:

                        superquadric = Superquadric(
                            object_ID=obj.tracking_id,
                            class_name=obj.object_class,
                            raw_data_1=rgb_msg,
                            bbox=obj.bbox,
                            raw_depth=depth_msg,
                            raw_mask=None,
                            camera_info=self.latest_camera_info
                        )

                        if superquadric:
                            all_models.append(superquadric.getAlignedPCD())
                            grasp_obj = Grasps(superquadric, 'front', True)
                            grasp_pose = grasp_obj.selectGrasps()

                            if grasp_pose:
                                pose_stamped = PoseStamped()
                                pose_stamped.header = rgb_msg.header
                                pose_stamped.pose = grasp_pose["pose"]
                                self.grasp_pose_pub.publish(pose_stamped)

                                point_1 = grasp_pose["point_1"]
                                point_2 = grasp_pose["point_2"]

                                marker_1 = self.publish_grasp_point_marker(point_1, rgb_msg.header.frame_id, marker_id=0, color=(1.0, 0.0, 0.0))
                                marker_2 = self.publish_grasp_point_marker(point_2, rgb_msg.header.frame_id, marker_id=1, color=(0.0, 0.0, 1.0))

                                self.marker_pub_1.publish(marker_1)
                                self.marker_pub_2.publish(marker_2)

                                rospy.loginfo("Published grasp pose and grasp points.")
                            else:
                                rospy.logwarn("No valid grasp pose found.")

                if all_models:
                    self.publish_combined_point_cloud(all_models, rgb_msg.header)
                else:
                    rospy.loginfo("No valid superquadric models to publish.")
            else:
                rospy.loginfo("No detected objects.")
        except Exception as e:
            rospy.logerr(f"Error in synchronized callback: {e}")

    def convert_o3d_to_ros_cloud(self, cloud_o3d, frame_id="head_rgbd_sensor_rgb_frame"):
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
