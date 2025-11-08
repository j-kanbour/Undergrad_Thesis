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

Sample all ponts on the target SQ
    ✅ For front- get closest point on closest sq	

    ✅ For top- get highest point on highest sq

    ✅ Z-axis points to centre of sq
    
    X-axis points along the long extent
    Y-axis points along the short extent

"""

import os
import sys
# import rospy
import numpy as np 
import open3d as o3d
import random

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
        

        self.primarySQ, self.grasp_point = self.SQFiltering(sq_list)


        self.selectedGrasps = self.generatePose(self.grasp_point, self.primarySQ)

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
                
                for sq in sq_closest:  # Iterate through each superquadric
                    sorted_extent = sorted(sq.getBBOXExtent(), reverse=True)

                    if sorted_extent[0] > self.grasp_width and sorted_extent[1] > self.grasp_width:
                        continue
                    # If we get here, this object is valid
                    sq_center = sq.getCenter()
                    return sq, sq_center
                
                # If no valid object found, return None
                return None, None

            elif self.orientation == 'top':
                # Choose the object with the highest vertical centre first.
                # If your camera/world is Z-up, set VERT = 2; if Y-up, set VERT = 1.
                VERT = 2
                sq_sorted = sorted(sq_list, key=lambda sq: sq.getCenter()[VERT], reverse=True)

                for sq in sq_sorted:
                    exts = sorted(sq.getBBOXExtent(), reverse=True)
                    if exts[0] > self.grasp_width and exts[1] > self.grasp_width:
                        continue

                    # Get mesh points and highest surface point
                    mesh_pts = np.asarray(sq.getSuperquadricMesh().points)
                    if mesh_pts.size == 0:
                        continue

                    idx = np.argmax(mesh_pts[:, VERT])   # topmost point along the vertical axis
                    highest_point = mesh_pts[idx]

                    # Move the grasp centre upward by 5 cm
                    adjusted_point = np.copy(highest_point)
                    adjusted_point[VERT] += 0.20

                    return sq, adjusted_point

                return None, None

            else:
                return None, None  # Or handle other cases appropriately
        
        except Exception as e:
            print(f"grasp [graspPointFiltering] Error: {e}")
            return None

    def generatePose(self, grasp_point, sq):
        """
        Generate an Open3D coordinate frame with orientation based on superquadric extents.
        """
        # Get superquadric properties
        sq_center = sq.getCenter()
        bbox_extent = sq.getBBOXExtent()
        init_pose = sq.getSQPose()
        object_center = self.object_center
        
        # Calculate orientation
        # a) Z-axis points towards object_center in x,y plane only
        target_point = np.array([object_center[0], object_center[1], grasp_point[2]])
        z_axis = target_point - np.array(grasp_point)
        z_axis = z_axis / np.linalg.norm(z_axis)  # Normalize
        
        # Find the shortest and longest extent axes
        extents = bbox_extent
        sorted_indices = np.argsort(extents)
        min_extent_idx = sorted_indices[0]  # Shortest
        max_extent_idx = sorted_indices[2]  # Longest
        
        # Get the axes in the superquadric's local frame
        shortest_axis_local = np.zeros(3)
        shortest_axis_local[min_extent_idx] = 1.0
        
        longest_axis_local = np.zeros(3)
        longest_axis_local[max_extent_idx] = 1.0
        
        # Transform to world frame using the pose rotation matrix
        rotation_matrix_sq = init_pose[:3, :3]
        
        shortest_axis_world = rotation_matrix_sq @ shortest_axis_local
        longest_axis_world = rotation_matrix_sq @ longest_axis_local
        
        # Project axes onto plane perpendicular to z_axis
        # Y-axis should align with shortest extent
        y_axis = shortest_axis_world - np.dot(shortest_axis_world, z_axis) * z_axis
        y_axis_norm = np.linalg.norm(y_axis)
        
        # X-axis should align with longest extent
        x_axis = longest_axis_world - np.dot(longest_axis_world, z_axis) * z_axis
        x_axis_norm = np.linalg.norm(x_axis)
        
        # Handle edge cases where projection is too small
        if y_axis_norm < 0.1 or x_axis_norm < 0.1:
            # Fallback: use the axis that has better projection
            if y_axis_norm > x_axis_norm:
                y_axis = y_axis / y_axis_norm
                x_axis = np.cross(y_axis, z_axis)
                x_axis = x_axis / np.linalg.norm(x_axis)
            else:
                x_axis = x_axis / x_axis_norm
                y_axis = np.cross(z_axis, x_axis)
                y_axis = y_axis / np.linalg.norm(y_axis)
        else:
            # Normalize both
            y_axis = y_axis / y_axis_norm
            x_axis = x_axis / x_axis_norm
            
            # Make sure they're orthogonal by adjusting x_axis
            x_axis = x_axis - np.dot(x_axis, y_axis) * y_axis
            x_axis = x_axis / np.linalg.norm(x_axis)
            
            # Ensure right-handed coordinate system
            if np.dot(np.cross(x_axis, y_axis), z_axis) < 0:
                x_axis = -x_axis
        
        # Construct rotation matrix [x_axis, y_axis, z_axis]
        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
        
        # Create coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1,  # Adjust this value based on your scale
            origin=[0, 0, 0]
        )
        
        # Apply rotation
        coordinate_frame.rotate(rotation_matrix, center=[0, 0, 0])
        
        # Translate to grasp point
        coordinate_frame.translate(grasp_point)
        
        return coordinate_frame
        
    def getGraspPoints(self):
        return self.graspPoints
    
    def getSelectedGrasps(self):
        return self.selectedGrasps