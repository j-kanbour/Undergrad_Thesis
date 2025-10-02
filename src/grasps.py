"""
    Grasp Generator

    author: Jayden Kanbour
    UNSW student_id: z5316799

    Description:    given a point cloud model of an object, this class will generate 
                    possible grasps points, using a series of criteria to determine
                    optimal grasp points

    Input:
        - Superquadric object
        - orientation: {top, top2, front, front-vertical}

    Output:
        - pose : center point of contact on the object surface, pointign away from the object
        - grasp point 1
        - grasp point 2 

"""

import os
import sys
import rospy
import numpy as np 
import open3d as o3d
import random
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R

module_path = os.environ.get("UNSW_WS")
sys.path.append(module_path + "/PLANNING/action_server/src/grasp_code")

import grasp_checks
class Grasps:
    def __init__(self, sq, frame_id, orientation=None, gripper_width = 0.5, gripper_depth = 0.5):
        #blinky: depth=0.0666, width=0.236

        self.print = lambda *args, **kwargs: print("Grasps:", *args, **kwargs)

        #extract necessary information from superquadric object
        #sq = sq.getSuperquadricAsPCD()
        
        # self.depth_masked = sq.getRawData()["masked_depth"]
        
        # #extracts camera info
        # self.camera_info = sq.getRawData()["camera_info"]
        # self.K = np.array(self.camera_info.K).reshape(3, 3)
        # self.fx = self.K[0, 0]
        # self.fy = self.K[1, 1]
        # self.cx = self.K[0, 2]
        # self.cy = self.K[1, 2]
        # self.w = self.camera_info.width
        # self.h = self.camera_info.height
        
        # self.depth_scale = 0.001 #??

        # self.object_pcd = sq.getPCD().getPCD()

        #generate and select best grasp
        self.primaryPoints, self.sq_pose = self.graspPointFiltering(sq, orientation, gripper_depth, gripper_width)
        self.selectedGrasps = self.generatPose(self.primaryPoints, frame_id, self.sq_pose)

    def extractPointsAlongAxis(self, sq, orientation, angle_tol_deg= 5.0, extent_threshold=0.236):
        """
        From a point cloud sq with normals, return points whose normals are parallel to
        the object x, y, or z axes defined by its oriented bounding box (either + or −).
        The result will exclude points on faces where both extents (other than the relevant axis) are larger than a threshold value (0.236).
        The result will be returned as an open3d.geometry.PointCloud.
        """

        # Pull arrays
        P = np.asarray(sq.points)            # (N, 3)
        N = np.asarray(sq.normals)           # (N, 3)

        if P.size == 0:
            return o3d.geometry.PointCloud()  # Return an empty PointCloud if no points

        # Ensure normals are unit length (guard against non-normalised inputs)
        n_norm = np.linalg.norm(N, axis=1, keepdims=True)
        n_norm[n_norm == 0] = 1.0
        N = N / n_norm

        # Object axes from OBB rotation
        obb = sq.get_oriented_bounding_box()
        R = obb.R  # 3x3
        #ex, ey, ez = R[:, 0], R[:, 1], R[:, 2]  # world-space unit axes for object x, y, z

        # Get OBB extents (half lengths in each direction)
        extents = np.array(2*obb.extent)  # [width, height, depth]
        
        # Angle test: |dot(n, axis)| >= cos(theta)
        c = np.cos(np.deg2rad(angle_tol_deg))
        dots = np.abs(N @ R)   # shape (N, 3): [|n·ex|, |n·ey|, |n·ez|]

        # Any axis match
        mask_any = (dots >= c).any(axis=1)
        all_idx = np.where(mask_any)[0]

        # Filter the points based on OBB extents condition (exclude those points)
        filtered_points = P[all_idx]
        filtered_normals = N[all_idx]

        # Initialize list to hold valid points
        valid_points = []

        # Check each point to see if it lies on a face where the other extents are smaller than the threshold
        for i, _ in enumerate(filtered_points):
            normal = filtered_normals[i]
            
            # If the point's normal is close to the x, y, or z axis, check the corresponding extents
            if orientation in ['front', None] and np.abs(normal[0]) > 0.5:  # X axis (normal aligned with X face)
                if extents[1] <= extent_threshold or extents[2] <= extent_threshold:  # Y and Z extents must be below threshold
                    valid_points.append(filtered_points[i])

            elif orientation in ['front', None] and np.abs(normal[1]) > 0.5:  # Y axis (normal aligned with Y face)
                if extents[0] <= extent_threshold or extents[2] <= extent_threshold:  # X and Z extents must be below threshold
                    valid_points.append(filtered_points[i])

            elif orientation in ['top', None] and normal[2] > 0.5:  # Z axis (normal aligned with Z face) positve only so it faces up
                if extents[0] <= extent_threshold or extents[1] <= extent_threshold:  # X and Y extents must be below threshold
                    valid_points.append(filtered_points[i])

        print(f"Valid points count: {len(valid_points)}")
        # Convert the valid points back into Open3D PointCloud object
        valid_pcd = o3d.geometry.PointCloud()
        valid_pcd.points = o3d.utility.Vector3dVector(np.array(valid_points))

        return valid_pcd, R

    def graspPointFiltering(self, sq_list, orientation=None, gripper_depth=0.0666, gripper_width=0.236):
        #sort sq_list by sq size (i.e. number of points)
        #perform extractPointsAlongAxis on it
            #if points found generate grasps for each point and procede to selecction
            #if not then move to next largest sq
            #if none then select center of largest sq as grasp point

        sq_list = sorted(sq_list, key=lambda x: len(x.points), reverse=True)
        
        for sq in sq_list:
            primary_points = self.extractPointsAlongAxis(sq, orientation, angle_tol_deg=5.0, extent_threshold=gripper_width)
            if len(primary_points.points) > 0:
                self.print(f"Found {len(primary_points.points)} primary points on superquadric with {len(sq.points)} points.")
                return random.choice(primary_points.points)
            else:
                self.print(f"No primary points found on superquadric with {len(sq.points)} points.")
        
        self.print("No primary points found on any superquadric. Defaulting to center of largest superquadric.")
        return sq_list[0].get_center()

    def generatePose(self, grasp_point, frame_id, sq_pose):
        """
        Generate a PoseStamped by projecting a pose onto a point.
        
        Args:
            grasp_point: Open3D point (numpy array [x, y, z])
            sq_pose: Original object pose containing rotation (R matrix or quaternion)
        
        Returns:
            PoseStamped with the point position and projected rotation
        """
        # Create PoseStamped message
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = frame_id  # Change to your frame
        pose_stamped.header.stamp = rospy.Time.now()  # or use your timestamp
        
        # Set position from grasp point
        pose_stamped.pose.position.x = grasp_point[0]
        pose_stamped.pose.position.y = grasp_point[1]
        pose_stamped.pose.position.z = grasp_point[2]
        
        # Handle rotation from sq_pose
        # Assuming sq_pose.R is a 3x3 rotation matrix
        if hasattr(sq_pose, 'R'):
            rotation_matrix = sq_pose.R
        else:
            # If sq_pose is already a rotation matrix
            rotation_matrix = sq_pose
        
        # Convert rotation matrix to quaternion
        scipy_rotation = R.from_matrix(rotation_matrix)
        quat = scipy_rotation.as_quat()  # Returns [x, y, z, w]
        
        # Set orientation
        pose_stamped.pose.orientation.x = quat[0]
        pose_stamped.pose.orientation.y = quat[1]
        pose_stamped.pose.orientation.z = quat[2]
        pose_stamped.pose.orientation.w = quat[3]
        
        return pose_stamped

    def getAllGrasps(self):
        return self.allGrasps
    
    def getSelectedGrasps(self):
        return self.selectedGrasps
    