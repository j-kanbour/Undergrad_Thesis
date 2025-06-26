#!/usr/bin/env python3.8
import rospy
import os
import sys
import numpy as np
import message_filters
# ✅ Added import
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from unsw_vision_msgs.msg import DetectionList
from superquadric import Superquadric
from grasps import Grasps
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped

# Extend path for util scripts
module_path = os.environ.get("UNSW_WS")
sys.path.append(module_path + "/src/utils/src")  # NOTE: Change to "/VISION/utils/src" on robot

class GraspGenerator():
    def __init__(self):
        rospy.init_node("grasp_generator_node")
        self.bridge = CvBridge()

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

        # ✅ Grasp pose publisher
        self.grasp_pose_pub = rospy.Publisher('/grasp_pose', PoseStamped, queue_size=10)

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
                    if obj.object_class not in []:  # Target objects

                        superquadric = Superquadric(
                            object_ID=obj.tracking_id,
                            class_name=obj.object_class,
                            input_type="RGBD STREAM",
                            raw_data_1=rgb_msg,
                            bbox=obj.bbox,
                            raw_depth=depth_msg,
                            raw_mask=None,
                            camera_info=self.latest_camera_info
                        )

                        if superquadric:
                            all_models.append(superquadric.getAlignedPCD())
                            # ✅ Generate grasp pose
                            grasp_obj = Grasps(superquadric)
                            grasp_pose = grasp_obj.selectGrasps()

                            if grasp_pose:
                                pose_stamped = PoseStamped()
                                pose_stamped.header = rgb_msg.header  # Copy frame + timestamp
                                pose_stamped.pose = grasp_pose
                                self.grasp_pose_pub.publish(pose_stamped)
                                rospy.loginfo("Published grasp pose.")
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
        print(colors)

        if points.shape[0] == 0:
            return None

        rgb_packed = (colors * 255).astype(np.uint8)
        rgb_packed = rgb_packed[:, 0] << 16 | rgb_packed[:, 1] << 8 | rgb_packed[:, 2]

        cloud_data = []
        for i in range(points.shape[0]):
            x, y, z = points[i]
            cloud_data.append([x, y, z])

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
                rospy.loginfo(f"Published point cloud for object")

if __name__ == "__main__":
    node = GraspGenerator()
    rospy.spin()
