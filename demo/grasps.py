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
import numpy as np 
import open3d as o3d

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


        self.selectedGrasps = self.generatePose(self.primarySQ, self.grasp_point)

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
                sq_sorted = sorted(sq_list, key=lambda sq: sq.getCenter()[VERT], reverse=False)

                for sq in sq_sorted:
                    exts = sorted(sq.getBBOXExtent(), reverse=True)
                    if exts[0] > self.grasp_width and exts[1] > self.grasp_width:
                        continue

                    # Get mesh points and highest surface point
                    mesh_pts = np.asarray(sq.getSuperquadricMesh().points)
                    if mesh_pts.size == 0:
                        continue

                    idx = np.argmax(mesh_pts[:, VERT])   # topmost point along the vertical axis
                    highest_point = sq.getCenter()

                    # Move the grasp centre upward by 5 cm
                    adjusted_point = np.copy(highest_point)
                    adjusted_point[VERT] += 0.0

                    return sq, adjusted_point

                return None, None

            else:
                return None, None  # Or handle other cases appropriately
        
        except Exception as e:
            print(f"grasp [graspPointFiltering] Error: {e}")
            return None


    def generatePose(self, sq, grasp_point):
        """
        Generate an Open3D coordinate frame at grasp_point.
        Z axis always points DOWN (0, 0, -1).
        X and Y axes are loose as long as they are orthonormal.
        """
        grasp_point = np.asarray(grasp_point, dtype=float)

        # --------- Z AXIS (DOWN) ---------
        z_axis = np.array([0.0, 0.0, 1.0], dtype=float)

        # --------- X AND Y AXES (LOOSE) ---------
        # Pick any vector not parallel to z
        tmp = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(np.dot(tmp, z_axis)) > 0.9:     # almost parallel
            tmp = np.array([0.0, 1.0, 0.0], dtype=float)

        # x ⟂ z
        x_axis = np.cross(tmp, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        x_axis 

        # y completes right-handed frame
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        # Rotation matrix
        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))

        # Create Open3D frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1,
            origin=[-0.09, -0.1, 0.03]
        )

        coordinate_frame.rotate(rotation_matrix, center=[0, 0, 0])
        coordinate_frame.translate(grasp_point)

        return coordinate_frame



    def getGraspPoints(self):
        return self.graspPoints
    
    def getSelectedGrasps(self):
        return self.selectedGrasps