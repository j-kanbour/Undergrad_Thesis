#!/usr/bin/env python3

"""
    Description: Rosnode used to generate grasp poses for quick and accurate grasping

    Author: Jayden Kanbour

    Subscribers:

        '/hsrb/head_rgbd_sensor/rgb/image_raw',                 Type = Image
        '/hsrb/head_rgbd_sensor/depth_registered/image_raw',    Type = Image
        '/hsrb/head_rgbd_sensor/depth_registered/camera_info',  Type = CameraInfor
        '/unsw_vision/detections/objects/positions',            Type = unsw_vision_msgs/DetectionList

    Publishers:

        '/grasp_pose'                                           Type = Pose ????

""" 

import rospy
import os
import sys
import numpy as np
import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from unsw_vision_msgs.msg import DetectionList
from superquadric import Superquadric
from grasps import Grasps
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge

module_path = os.environ.get("UNSW_WS")
sys.path.append(module_path + "/src/utils/src") #NOTE change to "/VISION/utils/src" on robot

class GraspGenerator():
    def __init__(self):
        rospy.init_node("grasp_generator_node")
        self.bridge = CvBridge()

        self.rgb_sub = message_filters.Subscriber('/hsrb/head_rgbd_sensor/rgb/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('/hsrb/head_rgbd_sensor/depth_registered/image_raw', Image)
        self.camera_info_sub = rospy.Subscriber('/hsrb/head_rgbd_sensor/depth_registered/camera_info', CameraInfo)
        
        #synchronise the rgb and depth camera messages
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1
        )
        self.sync.registerCallback(self.main_callback)

        #subscribe to unsw_vision_msgs/DetectionList messages and track last message
        #TODO messgae beth for an example of the new message
        self.detection_msg = []
        self.detection_sub = rospy.Subscriber('/unsw_vision/detections/objects/positions', DetectionList, callback=self.detections_callback)
        self.last_detection_time = rospy.Time.now()
        self.detection_timer = rospy.Timer(rospy.Duration(2.0), self.check_detection_timeout)

        rospy.loginfo('point_extraction_node')
    
    def detections_callback(self, detection_msg):
        self.last_detection_time = rospy.Time.now()
        self.detection_msg = detection_msg

    def check_detection_timeout(self, _):
        #if unsw_vision hasnt published in (2 seconds), remove tracked objects
        if (rospy.Time.now() - self.last_detection_time).to_sec() > 2.0: #TODO can change to a parameter 
            self.detections_callback([])

    def main_callback(self, rgb_msg, depth_msg):

        try:
            if self.detection_msg and self.detection_msg.objects:
                # Convert image messages to OpenCV format
                rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
                
                # Extract camera intrinsics
                camera_intrinsics = np.array([
                    [self.camera_info_msg.K[0], 0, self.camera_info_msg.K[2]],
                    [0, self.camera_info_msg.K[4], self.camera_info_msg.K[5]],
                    [0, 0, 1]
                ])

                all_models = []
                for obj in self.detection_msg.objects:
                    if obj.object_class in ['box']: #TODO set target objects as parameter, leave blank for all

                        """
                            TODO: Superquadric Generator

                            Input: 
                            - rgb image
                            - depth image
                            - mask
                            - camera intrinsics

                            Output:
                            - Superquadric estimation model of object

                            Function calls:
                            - superquadrics.py --> Superquadric Class Obejct

                        """

                        superquadric = Superquadric(
                            object_id = obj.tracking_id,
                            class_name = obj.object_class,
                            bbox = obj.bbox, #NOTE: TEMP till we have a mask
                            input_type = "RGBD STREAM",
                            raw_data_1 = rgb_image,
                            raw_depth = depth_image,
                            raw_mask = None, #TODO GET MASK
                            camera_info = camera_intrinsics
                        )

                        if superquadric:
                            all_models.append(superquadric)
                
                all_grasps = []
                for model in all_models:
                    """
                        TODO: Grasp Generator

                        Input:
                        - Superquadric PCD

                        Output:
                        - Grasp pose
                    
                    """
                    grasp = Grasps(model)

                    if grasp:
                        all_grasps.append(grasp)

                # If grasps exist, publish them
                #final_grasps = np.vstack(all_models) if all_models else np.empty((0, 3), dtype=np.float32)

            else:
                rospy.loginfo("No detected objects.")
                final_grasps = np.empty((0, 3), dtype=np.float32)

            self.publish_grasp(final_grasps, depth_msg.header)

        except Exception as e:
            rospy.logerr(f"Error in synchronized callback: {e}")

    def publish_grasp(self, grasps, header):

        """
            FIXME: I don't know whats going on here yet
        """

        # Use the incoming message's header for frame alignment
        header.frame_id = header.frame_id or "camera_rgb_optical_frame"

        if not grasps:
            rospy.logwarn("No grasps detected. Publishing empty msg.")

        # Create an empty point cloud if there are no points
        grasp_msg = None

        # Publish
        self.point_cloud_pub.publish(grasp_msg)
        rospy.loginfo("Published segmented object point cloud.")

if __name__ == "__main__":
    node = GraspGenerator()
    rospy.spin()


