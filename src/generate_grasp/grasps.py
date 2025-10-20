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

import rospy
import time
import numpy as np 
import open3d as o3d
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R

class Grasps:
    def __init__(self, sq_list, sq_poses, target_frame, orientation=None, grasp_width=0.5, debug=False):

        self.debug = debug
        self.target_frame = target_frame
        self.orientation = orientation
        self.grasp_width = grasp_width
        self.avgerate_centre = None

        #generate and select best grasp
        if debug:
            self.graspPoints = o3d.geometry.PointCloud()
        
        time_start = time.time()

        self.primaryPoints, self.sq_pose = self.graspPointFiltering(sq_list, sq_poses)
        time1 = time.time() - time_start

        self.selectedGrasps = self.generatePose(self.primaryPoints, self.target_frame, self.sq_pose)
        time2 = time.time() - time_start - time1

        if self.debug:
            print(f"Grasps: Grasp points generated within {time1:.3f}s")
            print(f"Grasps: Grasp pose generated within {time2:.3f}s")

    def graspPointFiltering(self, sq_list, sq_poses):
        """
            Sort superquadrics by size and find valid grasp points.
            Returns the selected grasp point AND the corresponding OBB rotation matrix.
        """

        try:
            if self.orientation == 'front' or self.orientation is None:
                # Sort by Euclidean distance in the XY–Z plane (closest object to camera)
                # Sort both lists together based on 3D distance from origin
                sq_pairs = sorted(zip(sq_list, sq_poses),
                                key=lambda pair: np.linalg.norm(pair[0].get_oriented_bounding_box().center[:3]))
                # Unzip back into separate lists
                sq_list, sq_poses = zip(*sq_pairs) if sq_pairs else ([], [])
                sq_list, sq_poses = list(sq_list), list(sq_poses)
                
                # Calculate average center
                self.average_centre = np.mean([sq.get_oriented_bounding_box().center for sq in sq_list], axis=0)
                
                return sq_list[0].get_center(), sq_poses[0]
                
            elif self.orientation == 'top':
                # Sort by Y-axis (highest object first)
                # Sort both lists together based on superquadric Y-coordinate
                sq_pairs = sorted(zip(sq_list, sq_poses),
                                key=lambda pair: pair[0].get_oriented_bounding_box().center[1],
                                reverse=False)
                # Unzip back into separate lists
                sq_list, sq_poses = zip(*sq_pairs) if sq_pairs else ([], [])
                sq_list, sq_poses = list(sq_list), list(sq_poses)
                
                # Calculate average center
                self.average_centre = np.mean([sq.get_oriented_bounding_box().center for sq in sq_list], axis=0)
                
                return sq_list[0].get_center(), sq_poses[0]
            
            return None, None
        
        except Exception as e:
            print(f"grasp [graspPointFiltering] Error: {e}")
            return None, None   

    def generatePose(self, grasp_point, frame_id, sq_pose):
        """
        Generate a PoseStamped by projecting a pose onto a point.
        """
        try:
            if sq_pose is None:
                print("grasp: Warning: No rotation matrix provided, using identity rotation")
                sq_pose = np.eye(3)

            # Create PoseStamped message
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = frame_id
            pose_stamped.header.stamp = rospy.Time.now()

            # Set position from grasp point
            pose_stamped.pose.position.x = float(grasp_point[0])
            pose_stamped.pose.position.y = float(grasp_point[1])
            pose_stamped.pose.position.z = float(grasp_point[2])

            # Calculate direction from grasp point to average center
            direction_to_center = self.average_centre - grasp_point
            direction_to_center = direction_to_center / np.linalg.norm(direction_to_center)  # Normalize

            # New Z-axis points toward average center
            new_z = direction_to_center

            if self.orientation == 'front' or self.orientation is None:
                # X-axis should point up (align with map Y-axis)
                desired_x = np.array([0, 1, 0])
                
                # Project desired X onto plane perpendicular to Z
                new_x = desired_x - np.dot(desired_x, new_z) * new_z
                new_x = new_x / np.linalg.norm(new_x)
                
                # Y-axis completes the right-handed coordinate system
                new_y = np.cross(new_z, new_x)
            else:
                # For other orientations, use original approach
                original_x = sq_pose[:, 0]
                
                # Reconstruct X-axis to be orthogonal to new Z (project and normalize)
                new_x = original_x - np.dot(original_x, new_z) * new_z
                new_x = new_x / np.linalg.norm(new_x)
                
                # Y-axis completes the right-handed coordinate system
                new_y = np.cross(new_z, new_x)

            # Construct modified rotation matrix
            modified_pose = np.column_stack([new_x, new_y, new_z])

            # Convert rotation matrix to quaternion
            scipy_rotation = R.from_matrix(modified_pose)
            quat = scipy_rotation.as_quat()  # Returns [x, y, z, w]

            # Set orientation
            pose_stamped.pose.orientation.x = float(quat[0])
            pose_stamped.pose.orientation.y = float(quat[1])
            pose_stamped.pose.orientation.z = float(quat[2])
            pose_stamped.pose.orientation.w = float(quat[3])

            if self.debug:
                print(f"grasp: Generated grasp pose at position: [{grasp_point[0]:.3f}, {grasp_point[1]:.3f}, {grasp_point[2]:.3f}]")
                print(f"grasp: Orientation (quaternion): [{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]")
                print(f"grasp: Z-axis pointing toward average center: [{direction_to_center[0]:.3f}, {direction_to_center[1]:.3f}, {direction_to_center[2]:.3f}]")

            return pose_stamped
        
        except Exception as e:      
            print(f"grasp [generatePose] Error: {e}")
            return None 

    def getGraspPoints(self):
        return self.graspPoints
    
    def getSelectedGrasps(self):
        return self.selectedGrasps