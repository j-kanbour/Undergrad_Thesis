#!/usr/bin/env python3
import os
import time
import rospy
import psutil
import tf2_ros
import numpy as np
import open3d as o3d
import tf2_geometry_msgs
from grasps import Grasps
from hsrb_interface import geometry
from superquadric import Superquadric
import sensor_msgs.point_cloud2 as pc2
from pointCloudData import PointCloudData
from geometry_msgs.msg import PoseStamped
from unsw_vision_msgs.msg import DetectionList
from tf.transformations import euler_from_quaternion
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField

process = psutil.Process(os.getpid())
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

# TF buffer & listener
tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)

class GraspGenerator():
    def __init__(self):
        rospy.init_node("grasp_generator_node")

        self.latest_rgb = None
        self.latest_depth = None
        self.camera_info = None

        #initalize subscribers
        self.rgb_sub = rospy.Subscriber('/hsrb/head_rgbd_sensor/rgb/image_raw', Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber('/hsrb/head_rgbd_sensor/depth_registered/image_raw', Image, self.depth_callback)
        self.camera_info_sub = rospy.Subscriber('/hsrb/head_rgbd_sensor/depth_registered/camera_info', CameraInfo, callback=self.camera_info_callback)
        self.detection_sub = rospy.Subscriber('/unsw_vision/detections/objects/positions', DetectionList, callback=self.main_callback)
        
        #initialise publishers
        self.grasp_pose_pub_camera = rospy.Publisher('/grasp_pose_camera', PoseStamped, queue_size=10)
        self.grasp_pose_pub_base = rospy.Publisher('/grasp_pose_base', PoseStamped, queue_size=10)
        self.grasp_pose_pub_hand = rospy.Publisher('/grasp_pose_hand', PoseStamped, queue_size=10)

        self.point_cloud_pub = rospy.Publisher('/superquadric', PointCloud2, queue_size=10)
    
        rospy.loginfo('GraspGenerator node initialized')

    # Callback functions to store latest camera messages
    def rgb_callback(self, msg):
        self.latest_rgb = msg
        
    def depth_callback(self, msg):
        self.latest_depth = msg
        
    def camera_info_callback(self, msg):
        self.camera_info = msg

    # Main called when vision detects an obejct
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
                    start_cpu, start_memory = get_performance_metrics()
                    rospy.loginfo(f"Starting processing for object ID: {obj.tracking_id}")

                    pointCloudObject = PointCloudData(
                        object_ID=obj.tracking_id,
                        raw_rgb=self.latest_rgb,
                        raw_depth=self.latest_depth,
                        mask=obj.mask,
                        camera_info=self.camera_info
                    )

                    # Checkpoint 1: PCD processing complete
                    self.print_checkpoint("Checkpoint 1: PCD Processing Complete", start_time, start_cpu, start_memory)
                    checkpoint1_time = time.time()
                    checkpoint1_cpu, checkpoint1_memory = get_performance_metrics()

                    superquadric = Superquadric(
                        object_ID=obj.tracking_id,
                        class_name=obj.object_class,
                        pcd=pointCloudObject
                    )

                    # Checkpoint 2: Superquadric processing complete
                    self.print_checkpoint("Checkpoint 2: Superquadric Processing Complete", checkpoint1_time, checkpoint1_cpu, checkpoint1_memory)
                    checkpoint2_time = time.time()
                    checkpoint2_cpu, checkpoint2_memory = get_performance_metrics()

                    graspPose = Grasps(
                        sq=superquadric,
                        orientation='front'
                    ).getSelectedGrasps()
                
                    # Checkpoint 3: Grasp Generation Complete
                    self.print_checkpoint("Checkpoint 3: Grasp Generation Complete", checkpoint2_time, checkpoint2_cpu, checkpoint2_memory)

                    #publish grasp pose and obejct model
                    if superquadric and graspPose:
                        
                        #publish superquadric model
                        superquadric = convert_o3d_to_ros_cloud(superquadric.getSuperquadricAsPCD())
                        self.point_cloud_pub.publish(superquadric)

                        #publish grasp in camera frame
                        self.grasp_pose_pub_camera.publish(graspPose)

                        #publish grasp in base frame
                        base_frame_pose = generate_base_pose(graspPose)[0]
                        self.grasp_pose_pub_base(base_frame_pose)

                        #publish grasp in hand frame
                        hand_frame_pose = generate_base_pose(graspPose)[1]
                        self.grasp_pose_pub_hand(hand_frame_pose)

                    else:
                        rospy.loginfo("No valid superquadric model or grasp pose to publish.")
                        
            else:
                rospy.loginfo("No detected objects.")
        except Exception as e:
            rospy.logerr(f"Error in synchronized callback: {e}")

if __name__ == "__main__":
    node = GraspGenerator()
    rospy.spin()

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
--------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------
--------------------------------------------HELPER FUNCTIONS--------------------------------------------
--------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

"""Pose Transformation Functions"""

def generate_hand_pose(grasp_pose):
    #transform the pose to be in hand frame
            # roll, pitch, yaw = euler_from_quaternion(quat_flipped)
        roll, pitch, yaw = euler_from_quaternion([grasp_pose.pose.orientation.x,
                                                    grasp_pose.pose.orientation.y,
                                                    grasp_pose.pose.orientation.z,
                                                    grasp_pose.pose.orientation.w])

        target_pose = geometry.pose(
            x=x,
            y=y,
            z=z,
            ei= roll,
            ej= pitch,
            ek= yaw
        )

        return target_pose

def generate_base_pose(grasp_pose):
    # transform from depth camera frame to base_footprint frame
    trans = tf_buffer.lookup_transform("base_footprint", pose.header.frame_id, rospy.Time.now(), rospy.Duration(1))
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

    poseStamped_msg = PoseStamped()
    poseStamped_msg.header.frame_id = "base_footprint"
    poseStamped_msg.pose.position.x = x
    poseStamped_msg.pose.position.y = y
    poseStamped_msg.pose.position.z = z
    poseStamped_msg.pose.orientation.x = quat_flipped[0]
    poseStamped_msg.pose.orientation.y = quat_flipped[1]
    poseStamped_msg.pose.orientation.z = quat_flipped[2]
    poseStamped_msg.pose.orientation.w = quat_flipped[3]


    # roll, pitch, yaw = euler_from_quaternion(quat_flipped)
    roll, pitch, yaw = euler_from_quaternion([poseStamped_msg.pose.orientation.x,
                                                poseStamped_msg.pose.orientation.y,
                                                poseStamped_msg.pose.orientation.z,
                                                poseStamped_msg.pose.orientation.w])

    target_pose = geometry.pose(
        x=x,
        y=y,
        z=z,
        ei= roll,
        ej= pitch,
        ek= yaw
    )

    return [poseStamped_msg, target_pose]

#convert superquadric from open3d point cloud to roscloud: debugging purposes
def convert_o3d_to_ros_cloud(cloud_o3d, frame_id="head_rgbd_sensor_rgb_frame"):
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
    
"""Performance Fucntions"""

def get_performance_metrics():
    """Get current CPU usage and memory usage"""
    cpu_percent = process.cpu_percent()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
    return cpu_percent, memory_mb

def print_checkpoint(checkpoint_name, start_time, start_memory):
    """Print performance metrics for a checkpoint"""
    current_time = time.time()
    current_cpu, current_memory = get_performance_metrics()
    
    elapsed_time = current_time - start_time
    cpu_usage = current_cpu  # Current CPU usage percentage
    memory_usage = current_memory
    
    rospy.loginfo(f"=== {checkpoint_name} ===")
    rospy.loginfo(f"Elapsed time: {elapsed_time:.3f} seconds")
    rospy.loginfo(f"CPU usage: {cpu_usage:.1f}%")
    rospy.loginfo(f"Memory usage: {memory_usage:.1f} MB")
    rospy.loginfo(f"Memory change: {memory_usage - start_memory:.1f} MB")
    rospy.loginfo("=" * 30)