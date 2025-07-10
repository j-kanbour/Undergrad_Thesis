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
        - pose : centre point of contact on the object surface, pointign away from the object
        - grasp point 1
        - grasp point 2 

"""

import open3d as o3d
import numpy as np
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_matrix
import grasp_checks


class Grasps:
    def __init__(self, sq, orientation=None, _debug=True):

        self.print = lambda *args, **kwargs: print("Grasps:", *args, **kwargs)
        self.orientation = orientation
        self._debug = _debug

        #extract necessary information from superquadric object
        self.superquadric = sq.getSuperquadricAsPCD()

        self.depth = sq.getRawData()["raw_depth"]
        self.depth_map = o3d.geometry.Image(self.depth.astype(np.uint16))
        
        self.mask = sq.getRawData()["raw_mask"]
        
        #extracts camera info
        self.camera_info = sq.getRawData()["camera_info"]
        self.K = np.array(self.camera_info.K).reshape(3, 3)
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        self.w = self.camera_info.width
        self.h = self.camera_info.height
        
        self.depth_scale = 0.001

        self.object_pcd = sq.getPCD().getPCD()

        #generate and select best grasp
        self.allGrasps = self.generateGrasps()
        self.selectedGrasps = self.selectGrasps()

    #generate num_grasps possible grasps
    def generateGrasps(self, num_grasps=10000, dist=1.0):

        try:
            #extracts array of points and normals from superquadric
            points = np.asarray(self.superquadric.points)
            normals = np.asarray(self.superquadric.normals)

            length_points = len(points)

            #list of possible grasps
            candidate_grasps = []

            #check all point pairs (skip 50 at a time as those points are too close to make a difference)
            for i in range(1, len(points), 50):
                
                #point pairs exist on opposite ends of array
                point1 = points[i]
                normal1 = normals[i]

                second_index = length_points - i
                point2 = points[second_index]
                normal2 = normals[second_index]

                #angle between line through points and xz plane (floor)
                vec = point2 - point1
                vec_norm = vec / np.linalg.norm(vec)

                vec_xz = vec.copy()
                vec_xz[1] = 0

                vec_xz_norm = vec_xz / np.linalg.norm(vec_xz)

                dot = np.dot(vec_norm, vec_xz_norm)
                dot = np.clip(dot, -1.0, 1.0)  # Ensure wi
                angle_rad = np.arccos(dot)
                angle_deg = np.degrees(angle_rad)

                #Create grasp pose
                grasp_pose = {
                    "index_i": i,
                    "index_j": second_index,
                    "point_i": point1.copy(),
                    "point_i_normals": normal1.copy(),
                    "point_j": point2.copy(),
                    "point_j_normals": normal2.copy(),
                    "angle_to_xz": angle_deg  
                }

                """
                    check for
                    - antipodal points (opposing normals)
                    - no collisions around grasp points
                    - grasps close enough to eachother (claw width)
                    - grasp points within reach (claw depth)
                    - check grasp corsses over visible surface
                    - check grasp aligns with 6DOF pose
                    - check orientation is correct
                        {top, top2, front, front-vertical}

                    Need to perform hiearchy map for these tests
                    - processing
                    - elemination strength
                """

                #grasp_checks.checkCollision(grasp_pose, self.depth_map,
                #                            self.object_pcd, self.mask,
                #                            self.camera_info, collision_threshold=0.05) and \
                #grasp_checks.checkOrientation(grasp_pose, self.orientation, angle_threshold=1) and \
                #grasp_checks.checkAcrossFace(grasp_pose, self.object_pcd, self.orientation, angle_threshold=1)
                if grasp_checks.checkGripper(grasp_pose, self.superquadric) and \
                grasp_checks.checkAntipodal(grasp_pose, normal_threshold=10) and \
                grasp_checks.checkCollision(grasp_pose, self.depth_map, self.mask,
                                            self.camera_info, collision_threshold=0.05):
                    candidate_grasps.append(grasp_pose)
                    
                if len(candidate_grasps) >= num_grasps:
                    break
            
            if self._debug:
                for i in candidate_grasps:
                    for e in i: 
                        print(f'{e}:{i[e]}')
                    print('\n')

            return candidate_grasps

        except Exception as e:
            print(f"[generateGrasps] Error: {e}")
            return None
        
    def selectGrasps(self):
        """
        Selects a potential grasp and returns grasp pose + surface points.
        
        Creates a pose vector that:
        - Originates from the midpoint between grasp points
        - Is perpendicular to the line between the two grasp points
        - For 'top' orientation: perpendicular to floor (pointing up/down)
        - For 'front' orientation: parallel to floor (horizontal)
        """
        try:
            for grasp in self.allGrasps:
                # Get midpoint between grasp points
                point_i = np.array(grasp["point_i"])
                point_j = np.array(grasp["point_j"])
                midpoint = (point_i + point_j) / 2.0
                
                # Grasp line direction (between the two grasp points)
                grasp_line = point_j - point_i
                grasp_line_normalized = grasp_line / np.linalg.norm(grasp_line)
                
                # Ground normal (assuming Z-axis is up)
                ground_normal = np.array([0, 0, 1])
                
                # Determine pose vector based on orientation
                if self.orientation in ['front', 'front-vertical']:
                    # FRONT: Horizontal approach (parallel to ground, perpendicular to grasp line)
                    
                    # Method 1: Try cross product with ground normal
                    pose_vector = np.cross(grasp_line_normalized, ground_normal)
                    
                    # Check if cross product is valid (grasp line not parallel to ground normal)
                    if np.linalg.norm(pose_vector) < 0.1:
                        # Grasp line is nearly vertical, use alternative horizontal direction
                        # Create horizontal vector perpendicular to grasp line
                        if abs(grasp_line_normalized[0]) < 0.9:  # Not parallel to X-axis
                            pose_vector = np.cross(grasp_line_normalized, np.array([1, 0, 0]))
                        else:  # Parallel to X-axis, use Y-axis
                            pose_vector = np.cross(grasp_line_normalized, np.array([0, 1, 0]))
                    
                    # Normalize the pose vector
                    pose_vector = pose_vector / np.linalg.norm(pose_vector)
                    
                    # Ensure it's horizontal (zero Z component)
                    pose_vector[2] = 0
                    pose_vector = pose_vector / np.linalg.norm(pose_vector)
                    
                elif self.orientation in ['top', 'top2']:
                    # TOP: Vertical approach (perpendicular to ground, perpendicular to grasp line)
                    
                    # Project grasp line onto horizontal plane (XY plane)
                    grasp_line_horizontal = np.array([grasp_line_normalized[0], grasp_line_normalized[1], 0])
                    
                    if np.linalg.norm(grasp_line_horizontal) > 0.1:
                        # Grasp line has horizontal component
                        # Create vertical vector perpendicular to horizontal projection
                        horizontal_perp = np.cross(grasp_line_horizontal, ground_normal)
                        horizontal_perp = horizontal_perp / np.linalg.norm(horizontal_perp)
                        
                        # The pose vector should be perpendicular to both grasp line and this horizontal perpendicular
                        pose_vector = np.cross(grasp_line_normalized, horizontal_perp)
                    else:
                        # Grasp line is purely vertical, approach can be any horizontal direction
                        pose_vector = np.array([1, 0, 0])  # Default to X-axis
                    
                    # Normalize the pose vector
                    pose_vector = pose_vector / np.linalg.norm(pose_vector)
                    
                    # For top grasps, ensure the pose vector has a significant vertical component
                    if abs(pose_vector[2]) < 0.5:
                        # If not sufficiently vertical, make it point upward
                        pose_vector = np.array([0, 0, 1])
                    
                else:
                    # Default case - use surface normal approach
                    # Average the normals at both grasp points
                    normal_i = np.array(grasp["point_i_normals"])
                    normal_j = np.array(grasp["point_j_normals"])
                    avg_normal = (normal_i + normal_j) / 2.0
                    pose_vector = avg_normal / np.linalg.norm(avg_normal)
                
                # Verify that pose vector is perpendicular to grasp line
                dot_product = np.dot(pose_vector, grasp_line_normalized)
                if abs(dot_product) > 0.1:  # Not sufficiently perpendicular
                    print(f"Warning: Pose vector not perpendicular to grasp line. Dot product: {dot_product}")
                    # Force perpendicularity by using Gram-Schmidt process
                    pose_vector = pose_vector - dot_product * grasp_line_normalized
                    pose_vector = pose_vector / np.linalg.norm(pose_vector)
                
                # Create pose as a vector (not a Pose message)
                pose = {
                    "origin": midpoint,
                    "vector": pose_vector,
                    "grasp_line_direction": grasp_line_normalized,  # Added for reference
                    "orientation_type": self.orientation
                }
                
                return {
                    "pose": pose,
                    "point_1": grasp["point_i"],
                    "point_2": grasp["point_j"]
                }
                

        except Exception as e:
            print(f"[selectGrasps] Error: {e}")
            return None
        
    def getAllGrasps(self):
        return self.allGrasps
    
    def getSelectedGrasps(self):
        return self.selectedGrasps
    