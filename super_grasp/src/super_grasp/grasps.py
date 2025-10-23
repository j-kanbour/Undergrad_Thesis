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
                sq_closest = list(sq_closest)[0]

                sq_points = sq_closest.getSuperquadricMesh().points
                points = np.asarray(sq_points)
                distances_xz = np.sqrt(points[:, 0]**2 + points[:, 2]**2)
                closest_index = np.argmin(distances_xz)
                closest_point = points[closest_index]

                return sq_closest, closest_point

            elif self.orientation == 'top':
                # Sort by Y-axis (highest object first)

                # Sort both lists together based on superquadric Y-coordinate
                sq_highest = sorted(sq_list,
                                key=lambda x: x.getCenter()[1],
                                reverse=False)

                # Unzip back into separate lists
                sq_highest = sq_list(sq_highest)[0]
                sq_points = sq_closest.getSuperquadricMesh().points
                points = np.asarray(sq_points)
                # Find the index of the point with maximum y value
                highest_index = np.argmax(points[:, 1])

                # Get the highest point
                highest_point = points[highest_index]

                return sq_highest, highest_point
            
            return None
        
        except Exception as e:
            print(f"grasp [graspPointFiltering] Error: {e}")
            return None

    def generatePose(self, sq, grasp_point, frame_id):
        """
        Generate a PoseStamped by projecting a pose onto a point.
        """
        
        sq_center = sq.getCenter()
        bbox_extent = sq.getBBOXExtent()
        init_pose = sq.getSQPose()
        object_center = self.object_center
        
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
            # a) Z-axis points towards object_center in x,y plane only
            # Create a target point at object_center's x,y but grasp_point's z
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
            # Extract rotation matrix from init_pose (assuming it's a 4x4 transformation matrix)
            rotation_matrix_sq = init_pose[:3, :3]
            
            shortest_axis_world = rotation_matrix_sq @ shortest_axis_local
            longest_axis_world = rotation_matrix_sq @ longest_axis_local
            
            # Project axes onto plane perpendicular to z_axis and assign to y and x
            # Y-axis should align with shortest extent
            y_axis = shortest_axis_world - np.dot(shortest_axis_world, z_axis) * z_axis
            y_axis_norm = np.linalg.norm(y_axis)
            
            # X-axis should align with longest extent
            x_axis = longest_axis_world - np.dot(longest_axis_world, z_axis) * z_axis
            x_axis_norm = np.linalg.norm(x_axis)
            
            # If one of the projections is too small, use cross product approach
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
                print(f"grasp: X-axis along longest extent: [{x_axis[0]:.3f}, {x_axis[1]:.3f}, {x_axis[2]:.3f}]")
                print(f"grasp: Extents [x,y,z]: [{extents[0]:.3f}, {extents[1]:.3f}, {extents[2]:.3f}]")
                print(f"grasp: Shortest axis index: {min_extent_idx}, Longest axis index: {max_extent_idx}")
            
            return pose_stamped
        
        except Exception as e:      
            print(f"grasp [generatePose] Error: {e}")
            return None
        
    def getGraspPoints(self):
        return self.graspPoints
    
    def getSelectedGrasps(self):
        return self.selectedGrasps