#!/usr/bin/env python3.8

import time
import rospy
import tf2_ros
import numpy as np
import open3d as o3d
import tf2_geometry_msgs
from generate_grasp.grasps import Grasps
from generate_grasp.superquadric import Superquadric
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker
from generate_grasp.pointCloudData import PointCloudData
from geometry_msgs.msg import PoseStamped
from unsw_vision_msgs.msg import DetectionList
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from generate_grasp.srv import grasp_msg, grasp_msgResponse  # Update with your package name

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

class GraspGeneratorService():
    def __init__(self):
        rospy.init_node("generate_grasp_service")
        
        # TF buffer & listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.latest_rgb = None
        self.latest_depth = None
        self.camera_info = None
        self.latest_detections = None

        # Load parameters from parameter server
        self.debug = bool(int(rospy.get_param("~debug", 0)))
        self.rgb_topic = rospy.get_param("~rgb_topic")
        self.depth_topic = rospy.get_param("~depth_topic")
        self.camera_info_topic = rospy.get_param("~camera_info_topic")
        self.detections_topic = rospy.get_param("~detections_topic")
        self.grasp_width = rospy.get_param("~grasp_width")

        # Initialize subscribers - these continuously cache the latest data
        self.rgb_sub = rospy.Subscriber(self.rgb_topic, Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber(self.depth_topic, Image, self.depth_callback)
        self.camera_info_sub = rospy.Subscriber(self.camera_info_topic, CameraInfo, callback=self.camera_info_callback)
        self.detection_sub = rospy.Subscriber(self.detections_topic, DetectionList, callback=self.detection_callback)
        
        # Initialize publishers for debugging
        if self.debug:
            self.grasp_pose_pub_hand= rospy.Publisher('grasp_pose_hand', PoseStamped, queue_size=10)
            self.grasp_pose_pub_base = rospy.Publisher('grasp_pose_base', PoseStamped, queue_size=10)
            self.grasp_pose_pub_camera = rospy.Publisher('grasp_pose_camera', PoseStamped, queue_size=10)
            self.superquadric_pub = rospy.Publisher('superquadric', PointCloud2, queue_size=10)
            self.point_cloud_pub = rospy.Publisher('extracted_point_cloud', PointCloud2, queue_size=10)
            self.superquadric_bbox_pub = rospy.Publisher('superquadric_bbox', Marker, queue_size=10)

        # Create the service
        self.service = rospy.Service('generate_grasp', grasp_msg, self.handle_generate_grasp)
        
        rospy.loginfo('GraspGenerator service initialized and ready')

    # Callback functions to cache latest messages
    def rgb_callback(self, msg):
        self.latest_rgb = msg
        
    def depth_callback(self, msg):
        self.latest_depth = msg
        
    def camera_info_callback(self, msg):
        self.camera_info = msg
    
    def detection_callback(self, msg):
        self.latest_detections = msg

    # Service handler - runs once per service call
    def handle_generate_grasp(self, req):
        response = grasp_msgResponse()
        
        try:
            # Check if we have the necessary data
            if self.latest_rgb is None or self.latest_depth is None or self.camera_info is None:
                response.success = False
                response.message = "Missing camera data. Ensure camera topics are publishing."
                rospy.logwarn(response.message)
                return response
            
            if self.latest_detections is None or not self.latest_detections.objects:
                response.success = False
                response.message = "No detections available."
                rospy.logwarn(response.message)
                return response

            if self.debug:
                rospy.loginfo("Starting grasp generation...")

            # Find the target object
            target_obj = None
            for obj in self.latest_detections.objects:
                # Check if this object matches the request criteria
                id_match = (req.target_object_id == -1 or obj.tracking_id == req.target_object_id)
                class_match = (req.target_object_class == "false" or obj.object_class == req.target_object_class)
                
                if id_match and class_match:
                    target_obj = obj
                    break
            
            if target_obj is None:
                response.success = False
                response.message = f"No object found matching criteria (ID: {req.target_object_id}, Class: {req.target_object_class})"
                rospy.logwarn(response.message)
                return response

            init_time = time.time()

            # Generate point cloud from rgb, depth and mask
            pcd = PointCloudData(        
                raw_rgb=self.latest_rgb,
                raw_depth=self.latest_depth,
                mask=target_obj.mask,
                camera_info=self.camera_info,
                nearest_neighbor=500,
                distance_thresh=0.005,
                semght_threshold=10,
                debug=self.debug
            )

            pcd_time = time.time()

            # Extract segments from point cloud
            cloudSegments = pcd.getCloudSegments()

            cloudSegment_time = time.time()

            superquadrics = []

            #build and publish all sqs for debugging
            if self.debug:
                allSQs = o3d.geometry.PointCloud()
                allClouds = o3d.geometry.PointCloud()

            #generate superquadric fits for each segment
            for segment in cloudSegments:
                sq = Superquadric(segment, downsample=30, debug=self.debug)
                superquadrics.append(sq)

                if self.debug:
                    sq_mesh = sq.getSuperquadricMesh()
                    allSQs += sq_mesh
                    allClouds += segment

            sq_time = time.time()

            #generate and select grasp from select superquadric
            graspPose = Grasps(
                superquadrics, 
                self.camera_info.header.frame_id, 
                object_center = pcd.getCenter(),
                orientation=req.orientation, 
                grasp_width=self.grasp_width, 
                debug=self.debug
            ).getSelectedGrasps()


            # Transform grasp to base frame
            base_frame_pose = self.camera_to_base_transform(graspPose)
            
            # Transform grasp to hand frame
            hand_frame_pose = self.camera_to_hand_transform(graspPose)

            grasp_time = time.time()

            if base_frame_pose is None or hand_frame_pose is None:
                response.success = False
                response.message = "Failed to transform grasp pose"
                rospy.logerr(response.message)
                return response

            # Populate response
            response.success = True
            response.message = "Grasp generated successfully"
            response.grasp_pose_base = base_frame_pose
            response.grasp_pose_hand = hand_frame_pose

            if self.debug:
                # Publish superquadric
                superquadric_model = self.convert_o3d_to_ros_cloud(allSQs)
                if superquadric_model:
                    self.superquadric_pub.publish(superquadric_model)

                # Publish segment point clouds
                pointcloud = self.convert_o3d_to_ros_cloud(allClouds)
                if pointcloud:
                    self.point_cloud_pub.publish(pointcloud)

                # Publish grasp in camera frame
                self.grasp_pose_pub_camera.publish(graspPose)

                self.grasp_pose_pub_hand.publish(hand_frame_pose)

                self.grasp_pose_pub_base.publish(base_frame_pose)

                rospy.loginfo(f"\n\n\n\n\
                    start time: {init_time} \n\
                    pcd_time: {pcd_time - init_time} \n\
                    segmentation_time: {cloudSegment_time - pcd_time} \n\
                    sq_time: {sq_time - cloudSegment_time} \n\
                    grasp_time: {grasp_time - sq_time} \n\
                    total time: {grasp_time - init_time} \n\
                    \n\n\n\n\n")

            rospy.loginfo("Grasp generation complete")
            return response
                        
        except Exception as e:
            response.success = False
            response.message = f"Error generating grasp: {str(e)}"
            rospy.logerr(response.message)
            return response

    """Pose Transformation Functions"""

    def camera_to_base_transform(self, grasp_pose):
        target_frame = "base_footprint"
        try:
            # Transform the pose frame
            T = self.tf_buffer.lookup_transform(target_frame, grasp_pose.header.frame_id, rospy.Time(0), rospy.Duration(0.2))
            pose = tf2_geometry_msgs.do_transform_pose(grasp_pose, T)
            
            # Calculate yaw to point towards the object (grasp pose position)
            dx = pose.pose.position.x
            dy = pose.pose.position.y
            yaw = np.arctan2(dy, dx)
            
            # Create horizontal orientation pointing at the object
            r = R.from_euler('xyz', [0, 0, yaw])
            
            # Get rotation matrix
            R_horiz = r.as_matrix()
            quat = r.as_quat()
            
            # Compute position: move 30 cm backwards along forward (x) axis
            forward_vector = R_horiz[:, 0]
            delta_pos = -0.5 * forward_vector
            new_pos = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]) + delta_pos
            x, y, z = new_pos
            
            base_pose = PoseStamped()
            base_pose.header.frame_id = "base_footprint"
            base_pose.pose.position.x = x
            base_pose.pose.position.y = y
            base_pose.pose.position.z = 0
            base_pose.pose.orientation.x = quat[0]
            base_pose.pose.orientation.y = quat[1]
            base_pose.pose.orientation.z = quat[2]
            base_pose.pose.orientation.w = quat[3]
            
            return base_pose
        
        except Exception as e:
            rospy.logwarn(f"camera to base transform {e}")
            return None

    def camera_to_hand_transform(self, grasp_pose):
        target_frame = "hand_palm_link"
        
        if grasp_pose is None:
            rospy.logerr("camera_to_hand_transform received None grasp_pose")
            return None
        
        try:
            T = self.tf_buffer.lookup_transform(
                target_frame, 
                grasp_pose.header.frame_id, 
                rospy.Time(0),
                rospy.Duration(2.0)
            )
            
            hand_pose = tf2_geometry_msgs.do_transform_pose(grasp_pose, T)
            hand_pose.header.stamp = T.header.stamp
            
            return hand_pose
            
        except Exception as e:
            rospy.logwarn(f'Camera to hand transform: {e}')
            return None

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

if __name__ == "__main__":
    service = GraspGeneratorService()
    rospy.spin()