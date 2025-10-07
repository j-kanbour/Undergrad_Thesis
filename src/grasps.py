"""
    Grasp Generator

    author: Jayden Kanbour
    UNSW student_id: z5316799

    # get grasp to match the direction its orientation (i.e. if top then rotate accordingly)
    # if orientation is none then search all points
    # isolate top from bottom, front from back by the plane with most pcd points closest to it (how??)
    # if no points found then default to center of largest superquadric

"""

import os
import sys
# import rospy
import numpy as np 
import open3d as o3d
import random
# from geometry_msgs.msg import PoseStamped
# from scipy.spatial.transform import Rotation as R

# module_path = os.environ.get("UNSW_WS")
# sys.path.append(module_path + "/PLANNING/action_server/src/grasp_code")

class Grasps:
    def __init__(self, sq, sq_pose, orientation=None, gripper_width = 0.5, gripper_depth = 0.5):
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
        self.graspPoints = o3d.geometry.PointCloud()
        self.primaryPoints, self.pose = self.graspPointFiltering(sq, sq_pose, orientation, gripper_depth, gripper_width)
        self.selectedGrasps = self.generatePose(self.primaryPoints, self.pose)

    def extractPointsAlongAxis(self, sq, pose, orientation, angle_tol_deg= 5.0, extent_threshold=0.236):
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

        # if orientation == 'front':
        #     #rotate R 90 degrees about x axis and y axis
        #     R = R @ np.array([[1, 0, 0],
        #                       [0, 0, -1],
        #                       [0, 1, 0]])
        # elif orientation == 'top':
        #     #rotate R 90 degrees about y axis and then 180 about x
        #     R = R @ np.array([[0, 0, 1],
        #                       [0, 1, 0],
        #                       [-1, 0, 0]])
        #ex, ey, ez = R[:, 0], R[:, 1], R[:, 2]  # world-space unit axes for object x, y, z

        # Get OBB extents (half lengths in each direction)
        extents = np.array(2*obb.extent)  # [width, height, depth]
        
        # Angle test: |dot(n, axis)| >= cos(theta)
        c = np.cos(np.deg2rad(angle_tol_deg))
        dots = np.abs(N @ pose)   # shape (N, 3): [|n·ex|, |n·ey|, |n·ez|]

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
            if (orientation == 'front') and np.abs(normal[0]) > 0.7:  # X axis (np.abs(normal al)igned with X face)
                print('front, x')
                if extents[1] < extent_threshold or extents[2] < extent_threshold:  # Y and Z extents must be below threshold
                    valid_points.append(filtered_points[i])
                if extents[0] < extent_threshold or extents[1] < extent_threshold:  # X and Y extents must be below threshold
                    valid_points.append(filtered_points[i])

            if (orientation == 'top') and normal[1] > 0.7:  # Y axis (np.abs(normal al)igned with Y face)
                print('side, y')
                if extents[0] < extent_threshold or extents[2] < extent_threshold:  # X and Z extents must be below threshold
                    valid_points.append(filtered_points[i])

        print(f"Valid points count: {len(valid_points)}")

        # Convert the valid points back into Open3D PointCloud object
        valid_pcd = o3d.geometry.PointCloud()
        if len(valid_points) > 0:
            valid_pcd.points = o3d.utility.Vector3dVector(np.array(valid_points))
            self.graspPoints += valid_pcd

        return valid_pcd

    def graspPointFiltering(self, sq_list, sq_pose, orientation=None, gripper_depth=0.0666, gripper_width=0.236):
        #sort sq_list by sq size (i.e. number of points)
        #perform extractPointsAlongAxis on it
            #if points found generate grasps for each point and procede to selecction
            #if not then move to next largest sq
            #if none then select center of largest sq as grasp point

        sq_list = sorted(sq_list, key=lambda x: len(x.points), reverse=True)
        
        for sq, pose in zip(sq_list, sq_pose):
            primary_points = self.extractPointsAlongAxis(sq, pose, orientation, angle_tol_deg=5.0, extent_threshold=gripper_width)
            if len(primary_points.points) > 0:
                self.print(f"Found {len(primary_points.points)} primary points on superquadric with {len(sq.points)} points.")
                                # DETERMINISTIC SELECTION: Use centroid of all primary points
                points_array = np.asarray(primary_points.points)
                selected_point = np.mean(points_array, axis=0)
                
                return selected_point, pose  # Return point and rotation
            else:
                self.print(f"No primary points found on superquadric with {len(sq.points)} points.")
        
        self.print("No primary points found on any superquadric. Defaulting to center of largest superquadric.")
        return sq_list[0].get_center(), sq_pose[0]  # Return center of largest sq and its rotation
    
    def generatePose(self, grasp_point, sq_pose):
        """
        Generate a coordinate frame mesh for visualization with Open3D.
        
        Args:
            grasp_point: Open3D point (numpy array [x, y, z])
            sq_pose: Original object pose containing rotation (R matrix or quaternion)
        
        Returns:
            Open3D TriangleMesh representing a coordinate frame at the grasp pose
        """
        # Handle rotation from sq_pose
        # Assuming sq_pose.R is a 3x3 rotation matrix
        rotation_matrix = sq_pose
        
        # Create a coordinate frame mesh for visualization
        # Size parameter controls the length of the axes
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1,  # Adjust this value based on your scale
            origin=[0, 0, 0]
        )
        
        # Apply rotation
        coordinate_frame.rotate(rotation_matrix, center=[0, 0, 0])
        
        # Apply translation to grasp point
        coordinate_frame.translate(grasp_point)
        
        return coordinate_frame

    # def generatePose(self, grasp_point, frame_id, sq_pose):
    #     """
    #     Generate a PoseStamped by projecting a pose onto a point.
        
    #     Args:
    #         grasp_point: Open3D point (numpy array [x, y, z])
    #         sq_pose: Original object pose containing rotation (R matrix or quaternion)
        
    #     Returns:
    #         PoseStamped with the point position and projected rotation
    #     """
    #     # Create PoseStamped message
    #     pose_stamped = PoseStamped()
    #     pose_stamped.header.frame_id = frame_id  # Change to your frame
    #     pose_stamped.header.stamp = rospy.Time.now()  # or use your timestamp
        
    #     # Set position from grasp point
    #     pose_stamped.pose.position.x = grasp_point[0]
    #     pose_stamped.pose.position.y = grasp_point[1]
    #     pose_stamped.pose.position.z = grasp_point[2]
        
    #     # Handle rotation from sq_pose
    #     # Assuming sq_pose.R is a 3x3 rotation matrix
    #     if hasattr(sq_pose, 'R'):
    #         rotation_matrix = sq_pose.R
    #     else:
    #         # If sq_pose is already a rotation matrix
    #         rotation_matrix = sq_pose
        
    #     # Convert rotation matrix to quaternion
    #     scipy_rotation = R.from_matrix(rotation_matrix)
    #     quat = scipy_rotation.as_quat()  # Returns [x, y, z, w]
        
    #     # Set orientation
    #     pose_stamped.pose.orientation.x = quat[0]
    #     pose_stamped.pose.orientation.y = quat[1]
    #     pose_stamped.pose.orientation.z = quat[2]
    #     pose_stamped.pose.orientation.w = quat[3]
        
    #     return pose_stamped

    def getGraspPoints(self):
        return self.graspPoints
    
    def getSelectedGrasps(self):
        return self.selectedGrasps
    