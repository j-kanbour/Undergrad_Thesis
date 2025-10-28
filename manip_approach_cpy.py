#! /usr/bin/env python3

import rospy
import actionlib
import hsrb_interface
from action_server.msg import ManipApproachAction, ManipApproachFeedback, ManipApproachResult
from generate_grasp.srv import grasp_msg, grasp_msgRequest
from scipy.spatial.transform import Rotation as R
import tf.transformations as tf_trans
from math import pi
import tf2_ros
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from hsrb_interface import geometry
import sys
import os
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import PoseStamped

module_path = os.environ.get("UNSW_WS")
sys.path.append(module_path + "/PLANNING/action_server/src")

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
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        self.grasp_pose_pub_hand= rospy.Publisher('grasp_pose_hand', PoseStamped, queue_size=10)
        self.grasp_pose_pub_base = rospy.Publisher('grasp_pose_base', PoseStamped, queue_size=10)

        #Vision Subscribers - Store latest messages instead of subscribers

        self._action_name = name
        self._as = actionlib.SimpleActionServer(self._action_name, ManipApproachAction, execute_cb=self.execute_cb, auto_start = False)
        self._as.start()
        rospy.loginfo("Action server manip-approach started")


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

    def pose_to_hsrb_geometry_pose(self, pose):
        """
        Convert a geometry_msgs/Pose to hsrb_interface geometry.Pose.
        
        Args:
            pose: geometry_msgs.msg.Pose
            
        Returns:
            hsrb_interface.geometry.Pose
        """
        try:
            # Extract position
            x = pose.position.x
            y = pose.position.y
            z = pose.position.z
            
            # Convert quaternion to Euler angles (roll, pitch, yaw)
            quaternion = (
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w
            )
            euler = tf_trans.euler_from_quaternion(quaternion, axes='sxyz')
            roll, pitch, yaw = euler
            
            # Create hsrb_interface geometry.Pose
            return geometry.pose(
                x=x, y=y, z=z,
                ei=roll, ej=pitch, ek=yaw,
                axes='sxyz'
            )
            
        except Exception as e:
            rospy.logerr(f"Error converting to hsrb geometry.Pose: {e}")
            return None

    def execute_cb(self, goal):

        r = rospy.Rate(1)
        feedback = ManipApproachFeedback()
        result = ManipApproachResult()
        # default to be true
        result.action_success=True

        feedback.feedback = 0  # Not started
        self._as.publish_feedback(feedback)

        direction = goal.direction
        goal_tracker_id = goal.object_id
        goal_object_class = goal.class_name

        #wait for grasp service
        rospy.wait_for_service('generate_grasp')

        try:
            generate_grasp = rospy.ServiceProxy('generate_grasp', grasp_msg)
            # Send msg to grasp service
            req = grasp_msgRequest()
            req.target_object_id = int(goal_tracker_id)
            req.target_object_class = goal_object_class
            req.orientation = direction
            
            # Retrieve grasp pose base and hand
            resp = generate_grasp(req)
            graspPose_base = resp.grasp_pose_base
            graspPose_hand = resp.grasp_pose_hand
            # self.grasp_pose_pub_hand.publish(graspPose_hand)

            # self.grasp_pose_pub_base.publish(graspPose_base)
            rospy.loginfo(f"Received grasp pose: {graspPose_hand, graspPose_base}")
            
            
        except rospy.ServiceException as e:
            rospy.logwarn("Service call failed: %s" % e)
            feedback.feedback = 2  # Failure
            self._as.publish_feedback(feedback)
            result.action_success = False
            result.forward_distance = 0
            self._as.set_aborted(result) # Important: exit after setting aborted

        # Actually move the robot
        try:
            # Set feedback to indicate we're starting
            feedback.feedback = 1  # In progress
            self._as.publish_feedback(feedback)
            
            self.open_gripper()
            
            # Move to neutral pose
            self._whole_body.move_to_neutral()
            
            # Move base to grasp position using whole_body
            try:
                # Move base only (not whole body)
                hsrb_pose = self.pose_to_hsrb_geometry_pose(graspPose_base.pose)
                self._whole_body.move_end_effector_pose(hsrb_pose, ref_frame_id='base_footprint', plan_only=False)
                
                rospy.sleep(1.0)
                rospy.loginfo("Base moved to grasp position")
                
            except Exception as e:
                rospy.logerr(f"Failed to move base: {e}")
                raise

            try:
                # Use the frame from the PoseStamped
                hsrb_pose = self.pose_to_hsrb_geometry_pose(graspPose_hand.pose)
                self._whole_body.move_end_effector_pose(hsrb_pose, ref_frame_id='hand_palm_link', plan_only=False)              
                
                rospy.sleep(1.0)
                rospy.loginfo("Hand moved to grasp position")
                
            except Exception as e:
                rospy.logerr(f"Failed to move hand: {e}")
                raise
            

            result.action_success = True
            result.forward_distance = 0.2
            self._as.set_succeeded(result)
            
        except Exception as e:
            rospy.logwarn("Failed moving body. {}".format(e))
            feedback.feedback = 2  # Failure
            self._as.publish_feedback(feedback)
            result.action_success = False
            result.forward_distance = 0
            self._as.set_aborted(result)
            return
        
if __name__ == '__main__':
    rospy.init_node('manip_approach')
    server = ManipApproach(rospy.get_name())
    rospy.spin()