#! /usr/bin/env python3

import rospy
import actionlib
import hsrb_interface
from hsrb_interface import geometry
import tf2_geometry_msgs
from action_server.msg import ManipApproachAction, ManipApproachFeedback, ManipApproachResult
from sensor_msgs.msg import Image, CameraInfo
from unsw_vision_msgs.msg import DetectionList
from scipy.spatial.transform import Rotation as R
from math import pi
import numpy as np
import tf2_ros

import math
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import PoseStamped
import sys
import os
module_path = os.environ.get("UNSW_WS")
sys.path.append(module_path + "/PLANNING/action_server/src")

from grasp_code.pointCloudData import PointCloudData
from grasp_code.superquadric import Superquadric
from grasp_code.grasps import Grasps

PALM_SIZE = 0.06
class ManipApproach(object):
    def __init__(self, name):
        # init the robot
        self._robot = hsrb_interface.Robot()
        self._whole_body = self._robot.get('whole_body')
        self._gripper = self._robot.get('gripper')

        self.move_base_action_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        if not self.move_base_action_client.wait_for_server(rospy.Duration(10)):
            rospy.logerr("move_base action server is not available")
            rospy.signal_shutdown("move_base action server is not available")
            return
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        #Vision Subscribers - Store latest messages instead of subscribers
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_camera_info = None
        self.latest_detections = None
        self.grasp_pose_pub = rospy.Publisher('grasp_pose', PoseStamped, queue_size=10)
        self.rgb_sub = rospy.Subscriber('/hsrb/head_rgbd_sensor/rgb/image_raw', Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber('/hsrb/head_rgbd_sensor/depth_registered/image_raw', Image, self.depth_callback)
        self.camera_info_sub = rospy.Subscriber('/hsrb/head_rgbd_sensor/depth_registered/camera_info', CameraInfo, callback=self.camera_info_callback)
        self.detection_sub = rospy.Subscriber('/unsw_vision/detections/objects/positions', DetectionList, callback=self.detections_callback)

        self._action_name = name
        self._as = actionlib.SimpleActionServer(self._action_name, ManipApproachAction, execute_cb=self.execute_cb, auto_start = False)
        self._as.start()
        rospy.loginfo("Action server manip-approach started")

    # Callback functions to store latest messages
    def rgb_callback(self, msg):
        self.latest_rgb = msg
        
    def depth_callback(self, msg):
        self.latest_depth = msg
        
    def camera_info_callback(self, msg):
        self.latest_camera_info = msg
        
    def detections_callback(self, msg):
        self.latest_detections = msg

    # open the gripper, can be extracted to a helper function of robot class
    def open_gripper(self):
        rospy.loginfo("opening gripper")
        self._gripper.command(1.2)
    # close the gripper, can be extracted to a helper function of robot class
    def close_gripper(self):
        self._gripper.apply_force(0.7)

    # move the arm lift joint to allign with the object height to avoid collision effectively
    def arm_joint_init(self, target_height):
        # when joint value is 0, it is 33cm from actual ground.
        joint_zero_rel_base = 0.33
        joint_values = {'arm_lift_joint': target_height - joint_zero_rel_base}
        self._whole_body.move_to_joint_positions(joint_values)

    def execute_cb(self, goal):
        r = rospy.Rate(1)
        feedback = ManipApproachFeedback()
        result = ManipApproachResult()
        # default to be true
        result.action_success=True

        feedback.feedback = 0  # Not started
        self._as.publish_feedback(feedback)

        direction = goal.direction
        target_point = goal.target_point
        goal_tracker_id = goal.object_id
        goal_object_class = goal.class_name
        

        feedback.feedback = 1  # Ongoing
        self._as.publish_feedback(feedback)

        try:
            trans = self.tfBuffer.lookup_transform("base_footprint", target_point.header.frame_id, rospy.Time.now(), rospy.Duration(2))
            transformed_point_stamped = tf2_geometry_msgs.do_transform_point(target_point, trans)
            angle_to_target = math.atan2(transformed_point_stamped.point.y, transformed_point_stamped.point.x)
            # print(angle_to_target)
            # rotate the robot to face the target and then another 90 degree make the right side of the robot facing the target
            # calc the new posestamp in base_footprint frame(0 + angle_to_target)
            goal_pose_rotation = tf2_geometry_msgs.PoseStamped()
            goal_pose_rotation.header.frame_id = "base_footprint"
            # calc the quanternion from ei ej ek use euler to quaternion
            q = quaternion_from_euler(0,0,angle_to_target)
            # print(q)
            goal_pose_rotation.pose.orientation.w = q[3]
            goal_pose_rotation.pose.orientation.z = q[2]
            goal_pose_rotation.pose.orientation.y = q[1]
            goal_pose_rotation.pose.orientation.x = q[0]
            trans2 = self.tfBuffer.lookup_transform("map", "base_footprint", rospy.Time.now(), rospy.Duration(2))
            transformed_pose_stamp_map = tf2_geometry_msgs.do_transform_pose(goal_pose_rotation, trans2)
            
            goal = MoveBaseGoal()
            goal.target_pose = transformed_pose_stamp_map
            # TODO: Uncommnet
            # self.move_base_action_client.send_goal_and_wait(goal)
        except Exception as e:
            rospy.loginfo("Failed to rotate to the object. {}".format(e))
            feedback.feedback = 2  # Failure
            self._as.publish_feedback(feedback)
            result.action_success = False
            result.forward_distance = 0
            self._as.set_aborted(result)
            return
        
        rospy.sleep(2)
        try:
            trans = self.tfBuffer.lookup_transform("base_footprint", target_point.header.frame_id, rospy.Time.now(), rospy.Duration(2))
            transformed_point_stamped = tf2_geometry_msgs.do_transform_point(target_point, trans)
            rospy.loginfo(transformed_point_stamped)
        except Exception as e:
            rospy.loginfo("Failed to transform point to footprint. {}".format(e))
            feedback.feedback = 2  # Failure
            self._as.publish_feedback(feedback)
            result.action_success = False
            result.forward_distance = 0
            self._as.set_aborted(result)
            return
        
        
        line_dis = 0
        
        if direction in ["top", "top2", "front", "front-vertical"]:

            #if only one instance of target object class, use that msg
            #else check the tracking id, trackign id will change if object moves out of frame
            current_object = None
            objects_detected = self.latest_detections.objects if self.latest_detections else []
            objects_in_class = {} #tracker_id : msg
            for obj in objects_detected:
                if obj.object_class == goal_object_class:
                    objects_in_class[obj.tracking_id] = obj
            if len(objects_in_class) == 1:
                current_object = list(objects_in_class.values())[0]  # Fixed: was getValue[0]
            else:
                current_object = objects_in_class.get(goal_tracker_id)
                
            if current_object is None:
                rospy.logwarn("No valid object found for grasping")
                feedback.feedback = 2  # Failure
                self._as.publish_feedback(feedback)
                result.action_success = False
                result.forward_distance = 0
                self._as.set_aborted(result)
                return

            pointCloudObject = PointCloudData(
                object_ID=current_object.tracking_id,
                raw_rgb=self.latest_rgb, 
                raw_depth=self.latest_depth,
                mask=current_object.mask,
                camera_info=self.latest_camera_info
            )

            superquadric = Superquadric(
                object_ID=current_object.tracking_id,
                class_name=current_object.object_class,
                pcd=pointCloudObject
            )

            pose = Grasps(
                sq=superquadric,  # Updated parameter name
                orientation=direction
            ).getSelectedGrasps()
            
            if pose is None:
                rospy.logwarn("No valid grasp pose found")
                feedback.feedback = 2  # Failure
                self._as.publish_feedback(feedback)
                result.action_success = False
                result.forward_distance = 0
                self._as.set_aborted(result)
                return

            # Convert the grasp pose to geometry.pose format for HSR
            # Note: You may need to adjust this conversion based on your specific pose format
            # pose = geometry.pose(
            #     x=selected_grasp_pose.x,
            #     y=selected_grasp_pose.y,
            #     z=selected_grasp_pose.z,
            #     ei=selected_grasp_pose.ei,
            #     ej=selected_grasp_pose.ej,
            #     ek=selected_grasp_pose.ek
            # )

            if direction in ["top", "top2"]:
                line_dis = pose.pose.position.z - transformed_point_stamped.point.z - PALM_SIZE
            elif direction in ["front", "front-vertical"]:  # Fixed typo: was "front-vrtical"
                line_dis = transformed_point_stamped.point.x - pose.pose.position.x - PALM_SIZE

        else:
            rospy.loginfo("Invalid direction")
            feedback.feedback = 2
            self._as.publish_feedback(feedback)
            result.action_success = False
            result.forward_distance = 0
            self._as.set_aborted(result)
            return
        # acually move the robot
        try:
            self.open_gripper()

            # transform from depth camera frame to base_footprint frame
            trans = self.tfBuffer.lookup_transform("base_footprint", pose.header.frame_id, rospy.Time.now(), rospy.Duration(1))
            pose = tf2_geometry_msgs.do_transform_pose(pose, trans)
            
            frame = u'hand_palm_link'
                
            # change pose to euler angles
            roll, pitch, yaw = euler_from_quaternion([pose.pose.orientation.x,
                                              pose.pose.orientation.y,
                                              pose.pose.orientation.z,
                                              pose.pose.orientation.w])
            
            r = R.from_euler('xyz', [roll, pitch, yaw])

            # Flipped orientation (180 deg around y-axis)
            R_orig = r.as_matrix()
            R_flip = R_orig @ R.from_euler('y', np.pi).as_matrix()
            quat_flipped = R.from_matrix(R_flip).as_quat()
            
            

            # Compute flipped position: move 20 cm backwards along flipped forward (z) axis
            forward_vector = R_flip[:, 0]  # Z axis of rotation matrix
            delta_pos = -0.3 * forward_vector
            new_pos = np.array([pose.pose.position.x , pose.pose.position.y, pose.pose.position.z]) + delta_pos
            x, y, z = new_pos

            debug_poseStamped_msg = PoseStamped()
            debug_poseStamped_msg.header.frame_id = "base_footprint"
            debug_poseStamped_msg.pose.position.x = x
            debug_poseStamped_msg.pose.position.y = y
            debug_poseStamped_msg.pose.position.z = z
            debug_poseStamped_msg.pose.orientation.x = quat_flipped[0]
            debug_poseStamped_msg.pose.orientation.y = quat_flipped[1]
            debug_poseStamped_msg.pose.orientation.z = quat_flipped[2]
            debug_poseStamped_msg.pose.orientation.w = quat_flipped[3]

            # roll, pitch, yaw = euler_from_quaternion(quat_flipped)
            roll, pitch, yaw = euler_from_quaternion([debug_poseStamped_msg.pose.orientation.x,
                                                      debug_poseStamped_msg.pose.orientation.y,
                                                      debug_poseStamped_msg.pose.orientation.z,
                                                      debug_poseStamped_msg.pose.orientation.w])

            
            self.grasp_pose_pub.publish(debug_poseStamped_msg)
            # move to that pose
            self._whole_body.move_to_neutral()
            # self.arm_joint_init(z)

            target_pose = geometry.pose(
                x=x,
                y=y,
                z=z,
                ei= roll,
                ej= pitch,
                ek= yaw
            )
            
            
            
            result.action_success = True
            result.forward_distance = 0.3
            self._as.set_succeeded(result)

            self._whole_body.move_end_effector_pose(target_pose,
                                                    "base_footprint")
            
            self._whole_body.move_end_effector_pose(geometry.pose(), "hand_palm_link_2")
            
            # move it to allign better
            # To overcome the tf misallign for hand plam link
            # THIS IS A HACK NOW NEED TO BE REVISED LATER
            if direction in ["top", "top2"]:
                line_traj = (-1, 0, 0)
                l =  0.03
                
                self._whole_body.end_effector_ffector_by_line(line_traj,l)
            # return the value if success, otherwise the error will be captured and false will be returned
            result.action_success = True
            result.forward_distance = line_dis
            self._as.set_succeeded(result)

            # set the feedback to not started
            feedback.feedback = 0  # Not started
            self._as.publish_feedback(feedback)
            return
        except Exception as e:
            rospy.logwarn("Failed moving body. {}".format(e))
            self._as.set_aborted(result)
            return

if __name__ == '__main__':
    rospy.init_node('manip_approach')
    server = ManipApproach(rospy.get_name())
    rospy.spin()