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
from tf.transformations import quaternion_from_matrix

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

        self.primarySQ, self.grasp_point = self.SQFiltering(sq_list)
        time1 = time.time() - time_start


        self.selectedGrasps = self.generatePose(self.primarySQ, self.grasp_point, self.target_frame)
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
        
    def generatePose(self, sq, grasp_point, frame_id):
        """
        Generate a PoseStamped by projecting a pose onto a point.
        """
        bbox_extent = sq.getBBOXExtent()
        
        try:
            # Create PoseStamped message
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = frame_id
            pose_stamped.header.stamp = rospy.Time.now()
            
            # Set position from grasp point
            pose_stamped.pose.position.x = float(grasp_point[0])
            pose_stamped.pose.position.y = float(grasp_point[1])
            pose_stamped.pose.position.z = float(grasp_point[2])
            
            # orientation z : points towards z AXIS of object_center
            # orientation y : points across shortest extent of bbox_extent
            # orientation x : remaining point
            # important note, the above orientations are in hierarchy
            
            # Z-axis is perpendicular to the radial line, pointing upward (world Z direction)
            # This is perpendicular to the line connecting the points in XY plane
            if self.orientation == 'front' or self.orientation == None:
                z_axis = np.array([0.0, 0.0, 1.0])
            else:
                z_axis = np.array([0.0,1.0,0.0])
            
            
            # Find shortest extent of bounding box for y-axis direction
            extents = np.array(bbox_extent)
            min_extent_idx = np.argmin(extents)
            
            # Create a temporary y-axis candidate along the shortest extent direction
            y_axis_candidate = np.zeros(3)
            y_axis_candidate[min_extent_idx] = 1.0
            
            # Make y-axis orthogonal to z-axis using Gram-Schmidt
            y_axis = y_axis_candidate - np.dot(y_axis_candidate, z_axis) * z_axis
            y_axis_norm = np.linalg.norm(y_axis)
            
            # Handle case where y_axis_candidate is parallel to z_axis
            if y_axis_norm < 1e-6:
                # Choose a different axis
                alt_idx = (min_extent_idx + 1) % 3
                y_axis_candidate = np.zeros(3)
                y_axis_candidate[alt_idx] = 1.0
                y_axis = y_axis_candidate - np.dot(y_axis_candidate, z_axis) * z_axis
                y_axis_norm = np.linalg.norm(y_axis)
            
            y_axis = y_axis / y_axis_norm
            
            # Calculate x-axis as cross product (completes right-handed coordinate system)
            x_axis = np.cross(y_axis, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)
            
            # Build rotation matrix from axes
            rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
            
            # Create 4x4 homogeneous transformation matrix
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            
            quaternion = quaternion_from_matrix(transform_matrix)
            
            # Set orientation
            pose_stamped.pose.orientation.x = quaternion[0]
            pose_stamped.pose.orientation.y = quaternion[1]
            pose_stamped.pose.orientation.z = quaternion[2]
            pose_stamped.pose.orientation.w = quaternion[3]
            
            return pose_stamped
            
        except Exception as e:
            rospy.logerr(f"Error generating pose: {e}")
            return None
        
    def getGraspPoints(self):
        return self.graspPoints
    
    def getSelectedGrasps(self):
        return self.selectedGrasps
    