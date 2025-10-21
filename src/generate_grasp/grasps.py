#!/usr/bin/env python3.8

"""
    grasps: grasp point generation and selection from superquadric fits

    Input:
        sq: point cloud of the superquadric to fit superquadric to
        sq_poses: list of 3x3 rotation matrices of the superquadrics
        target_frame: frame id for the generated PoseStamped
        orientation: 'front' or 'top' for selecting grasp points based on object orientation
        gripper_width: width of the gripper for filtering grasp points
    
    Output:
        selectedGrasps: PoseStamped of the selected grasp point
        graspPoints: point cloud of all potential grasp points

    Developed by: Jayden Kanbour as part of undergraduate thesis for UNSW Computer Science and Engineering
    Date: 26th November 2025
    Email: jkanbour1@gmail.com
    UNSW Student Id: z5316799

"""

"""
Notes, 
✅ sq-chosen depends on the orientation (front == closest, top == highest)

✅ z-axis align to the get the center of the object
❌ x-axis points along the long extent
❌ y-axis points along the short 'graspabale axis'

this way were focusing on possible grasps opposed to optimal

may need to re-configure the sq_list message passed to this function
such that instead of sq_list [open3d pcd] and sq_pose
just pass the sq object you created containg functions 
- get centroid (used to place point)
- get pose (not used)
- get extent (used with axis to orientate pose)
- get axis (used with extent to orientate pose)
"""

import rospy
import time
import numpy as np 
import open3d as o3d
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R

class Grasps:
    def __init__(self, sq_list, target_frame, object_center, orientation=None, grasp_width=0.5, debug=False):

        self.debug = debug
        self.target_frame = target_frame
        self.orientation = orientation
        self.grasp_width = grasp_width
        self.object_center = object_center

        #generate and select best grasp
        if debug:
            self.graspPoints = o3d.geometry.PointCloud()
        
        time_start = time.time()

        self.primarySQ = self.SQFiltering(sq_list)
        time1 = time.time() - time_start


        self.selectedGrasps = self.generatePose(self.primarySQ, self.target_frame)
        time2 = time.time() - time_start - time1

        if self.debug:
            print(f"Grasps: Grasp points generated within {time1:.3f}s")
            print(f"Grasps: Grasp pose generated within {time2:.3f}s")

    def SQFiltering(self, sq_list):
        """
            Sort superquadrics by size and find valid grasp points.
            Returns the selected grasp point AND the corresponding OBB rotation matrix.
        """

        try:
            # Sort sq_list by their position relative to the camera origin (closest first)
            if self.orientation == 'front' or self.orientation is None:
                # Sort by Euclidean distance in the XY–Z plane (closest object to camera)
                # Sort both lists together based on 3D distance from origin
                sq_closest = sorted(sq_list, key=lambda x: np.linalg.norm(x.getCenter()[:3]))

                # Unzip back into separate lists
                sq_closest = list(sq_closest)

                return sq_closest[0]

            elif self.orientation == 'top':
                # Sort by Y-axis (highest object first)

                # Sort both lists together based on superquadric Y-coordinate
                sq_highest = sorted(sq_list,
                                key=lambda x: x.getCenter()[1],
                                reverse=False)

                # Unzip back into separate lists
                sq_list(sq_highest)

                return sq_highest[0]
            
            return None
        
        except Exception as e:
            print(f"grasp [graspPointFiltering] Error: {e}")
            return None

    def generatePose(self, sq, frame_id):
        """
        Generate a PoseStamped by projecting a pose onto a point.
        """
        
        grasp_point = sq.getCenter()
        bbox_extent = sq.getBBOXExtent()
        init_pose = sq.getSQPose()
        object_center = self.object_center.flatten()
        
        try:
            # Create PoseStamped message
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = frame_id
            pose_stamped.header.stamp = rospy.Time.now()
            
            # Set position from grasp point
            pose_stamped.pose.position.x = float(grasp_point[0])
            pose_stamped.pose.position.y = float(grasp_point[1])
            pose_stamped.pose.position.z = float(grasp_point[2])
            
            # Calculate orientation
            # a) Z-axis points towards object_center
            z_axis = object_center - np.array(grasp_point)
            z_axis = z_axis / np.linalg.norm(z_axis)  # Normalize
            
            # Find the shortest extent axis
            extents = bbox_extent
            min_extent_idx = np.argmin(extents)
            
            # Get the axis corresponding to the shortest extent in the superquadric frame
            shortest_axis_local = np.zeros(3)
            shortest_axis_local[min_extent_idx] = 1.0
            
            # Transform to world frame
            shortest_axis_world = init_pose @ shortest_axis_local
            
            # Make y_axis perpendicular to z_axis
            # Project shortest_axis onto plane perpendicular to z_axis
            y_axis = shortest_axis_world - np.dot(shortest_axis_world, z_axis) * z_axis
            
            # # If y_axis is too small (shortest axis is parallel to z_axis), use alternative
            # if np.linalg.norm(y_axis) < 0.001:
            #     # Use any perpendicular vector
            #     if abs(z_axis[0]) < 0.9:
            #         y_axis = np.cross(z_axis, np.array([1, 0, 0]))
            #     else:
            #         y_axis = np.cross(z_axis, np.array([0, 1, 0]))
            
            y_axis = y_axis / np.linalg.norm(y_axis)  # Normalize
            
            # Calculate x_axis to complete right-handed coordinate system
            x_axis = np.cross(y_axis, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)  # Normalize
            
            # Construct rotation matrix [x_axis, y_axis, z_axis]
            rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
            
            # Convert rotation matrix to quaternion
            scipy_rotation = R.from_matrix(rotation_matrix)
            quat = scipy_rotation.as_quat()  # Returns [x, y, z, w]
            
            # Set orientation
            pose_stamped.pose.orientation.x = float(quat[0])
            pose_stamped.pose.orientation.y = float(quat[1])
            pose_stamped.pose.orientation.z = float(quat[2])
            pose_stamped.pose.orientation.w = float(quat[3])
            
            if self.debug:
                print(f"grasp: Generated grasp pose at position: [{grasp_point[0]:.3f}, {grasp_point[1]:.3f}, {grasp_point[2]:.3f}]")
                print(f"grasp: Orientation (quaternion): [{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]")
                print(f"grasp: Z-axis pointing to object center: [{z_axis[0]:.3f}, {z_axis[1]:.3f}, {z_axis[2]:.3f}]")
                print(f"grasp: Y-axis along shortest extent: [{y_axis[0]:.3f}, {y_axis[1]:.3f}, {y_axis[2]:.3f}]")
            
            return pose_stamped
        
        except Exception as e:      
            print(f"grasp [generatePose] Error: {e}")
            return None
        
    def getGraspPoints(self):
        return self.graspPoints
    
    def getSelectedGrasps(self):
        return self.selectedGrasps