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
    def __init__(self, sq, sq_poses, target_frame, orientation=None, gripper_width = 0.5, debug=False):

        self.debug = debug
        self.target_frame = target_frame
        self.orientation = orientation
        self.gripper_width = gripper_width

        #generate and select best grasp
        if debug:
            self.graspPoints = o3d.geometry.PointCloud()
        
        time_start = time.time()

        self.primaryPoints, self.sq_pose = self.graspPointFiltering(sq, sq_poses, self.orientation, self.gripper_width)
        time1 = time.time() - time_start

        self.selectedGrasps = self.generatePose(self.primaryPoints, self.target_frame, self.sq_pose)
        time2 = time.time() - time_start - time1

        if self.debug:
            print(f"Grasps: Grasp points generated within {time1:.3f}s")
            print(f"Grasps: Grasp pose generated within {time2:.3f}s")

    def extractPointsAlongAxis(self, sq, pose, orientation, grasp_width=0.236):

        """
            Extract points from the superquadric point cloud whose normals align with the specified axis.
            Additionally, remove points which width exceed the gripper width.
            Returns a point cloud of valid grasp points and the OBB rotation matrix.
        """
        try:
            # extract points and normals
            P = np.asarray(sq.points)            
            N = np.asarray(sq.normals)     

            if P.size == 0:
                return o3d.geometry.PointCloud(), None  # Return empty PointCloud and None for rotation

            # Ensure normals are unit length (guard against non-normalised inputs)
            n_norm = np.linalg.norm(N, axis=1, keepdims=True)
            n_norm[n_norm == 0] = 1.0
            N = N / n_norm

            # Object axes from OBB rotation
            obb = sq.get_oriented_bounding_box()
            obb_center = obb.center  

            # Get OBB extents (full lengths in each direction)
            extents = np.array(obb.extent) 

            if self.debug:
                print(f'grasp: {pose}')
                print(f"OBB Center: {obb_center}")
                print(f"OBB Extents: {extents}")
                print(f"OBB Rotation:\n{pose}")

            valid_points = []

            if orientation == 'front' or orientation is None: 
                valid_points.append(obb_center)
            else:
            # Check each point to see if it lies on a face where the other extents are smaller than the threshold
                for i in range(len(P)):
                    normal = N[i]
                    # 'top' face: y side
                    if (orientation == 'top') and (normal[1] > 0.5) and (P[i][1] <= obb_center[1]):
                        if (extents[0] < grasp_width) or (extents[2] < grasp_width):
                            valid_points.append(P[i])

            if self.debug:
                print(f"grasps: Valid points count: {len(valid_points)}")

            # Convert the valid points back into Open3D PointCloud object
            valid_pcd = o3d.geometry.PointCloud()
            if len(valid_points) > 0:
                valid_pcd.points = o3d.utility.Vector3dVector(np.array(valid_points))

                if self.debug:
                    self.graspPoints += valid_pcd

            return valid_pcd  # Return both point cloud and rotation matrix
        
        except Exception as e:
            print(f"grasp [extractPointsAlongAxis] Error: {e}")
            return o3d.geometry.PointCloud(), None

    def graspPointFiltering(self, sq_list, sq_poses, orientation=None, gripper_width=0.236):
        """
            Sort superquadrics by size and find valid grasp points.
            Returns the selected grasp point AND the corresponding OBB rotation matrix.
        """

        try:
            # Sort sq_list by their position relative to the camera origin (closest first)
            if orientation == 'front':
                # Sort by Euclidean distance in the XY–Z plane (closest object to camera)
                sq_list = sorted(sq_list,
                    key=lambda sq: np.linalg.norm(sq.get_oriented_bounding_box().center[:3])  # full 3D distance to origin
                )

            elif orientation == 'top':
                # Sort by Y-axis (highest object first)
                sq_list = sorted(sq_list,
                    key=lambda sq: sq.get_oriented_bounding_box().center[1],
                    reverse=True
                )

            for sq, pose in zip(sq_list, sq_poses):
                primary_points = self.extractPointsAlongAxis(sq, pose, orientation, gripper_width)
                
                if len(primary_points.points) > 0:
                    if self.debug:
                        print(f"grasp: Found {len(primary_points.points)} primary points on superquadric with {len(sq.points)} points.")
                    
                    # Use mean of all primary points
                    points_array = np.asarray(primary_points.points)
                    selected_point = np.mean(points_array, axis=0)
                    
                    # Return point and rotation
                    return selected_point, pose  
                else:
                    print(f"grasp: No primary points found on superquadric with {len(sq.points)} points.")
            
            # Fallback: use center of largest superquadric
            print("grasp: No primary points found on any superquadric. Defaulting to center of largest superquadric.")
            largest_sq = sq_list[0]
            
            # Get the OBB rotation for the fallback case too
            obb = largest_sq.get_oriented_bounding_box()
            fallback_rotation = obb.R
            
            return largest_sq.get_center(), fallback_rotation
        
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
            
            # Convert rotation matrix to quaternion and apply 180° rotation about Y-axis
            scipy_rotation = R.from_matrix(sq_pose)
            
            # Create 180° rotation about Y-axis
            rotation_y_180 = R.from_euler('y', 180, degrees=True)
            
            # Apply the rotation: new_rotation = original * y_rotation
            final_rotation = scipy_rotation * rotation_y_180
            
            quat = final_rotation.as_quat()  # Returns [x, y, z, w]
            
            # Set orientation
            pose_stamped.pose.orientation.x = float(quat[0])
            pose_stamped.pose.orientation.y = float(quat[1])
            pose_stamped.pose.orientation.z = float(quat[2])
            pose_stamped.pose.orientation.w = float(quat[3])
            
            if self.debug:
                print(f"grasp: Generated grasp pose at position: [{grasp_point[0]:.3f}, {grasp_point[1]:.3f}, {grasp_point[2]:.3f}]")
                print(f"grasp: Orientation (quaternion): [{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]")
            
            return pose_stamped
        
        except Exception as e:      
            print(f"grasp [generatePose] Error: {e}")
            return None 

    def getGraspPoints(self):
        return self.graspPoints
    
    def getSelectedGrasps(self):
        return self.selectedGrasps