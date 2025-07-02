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
    def __init__(self, superquadric, orientation=None, _debug=True):

        self.print = lambda *args, **kwargs: print("Grasps:", *args, **kwargs)
        self.orientation = orientation
        self._debug = _debug

        #extract necessary information from superquadric object
        self.superquadric = superquadric.getAlignedPCD()

        self.depth = superquadric.getPCD().getRawData()["raw_depth"]
        self.depth_map = o3d.geometry.Image(self.depth.astype(np.uint16))
        
        self.mask = superquadric.getPCD().getRawData()["raw_mask"]
        
        #extracts camera info
        self.camera_info = superquadric.getPCD().getRawData()["camera_info"]
        self.K = np.array(self.camera_info.K).reshape(3, 3)
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        self.w = self.camera_info.width
        self.h = self.camera_info.height
        
        self.depth_scale = 0.001

        self.object_pcd = superquadric.getPCD().getPCD()

        #generate and select best grasp
        self.allGrasps = self.generateGrasps()
        self.selectedGrasps = self.selectGrasps()

    #generate num_grasps possible grasps
    def generateGrasps(self, num_grasps=10000, dist=1.0):

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
            grasp_checks.checkAntipodal(grasp_pose, normal_threshold=10):
                candidate_grasps.append(grasp_pose)
                
            if len(candidate_grasps) >= num_grasps:
                break
        
        if self._debug:
            for i in candidate_grasps:
                for e in i: 
                    print(f'{e}:{i[e]}')
                print('\n')

        return candidate_grasps

    def selectGrasps(self):
        """
        Selects a potential grasp and returns grasp pose + surface points.
        """
        for grasp in self.allGrasps:
            # Get midpoint between grasp points
            midpoint = (grasp["point_i"] + grasp["point_j"]) / 2.0

            # Approach direction (z-axis of end-effector)
            approach = grasp["point_j"] - grasp["point_i"]
            approach /= np.linalg.norm(approach)

            # Create a fake up vector (e.g. camera's Y axis), then compute orthogonal axes
            fake_up = np.array([0, 1, 0])
            if abs(np.dot(fake_up, approach)) > 0.95:
                fake_up = np.array([1, 0, 0])  # fallback if aligned

            # Compute axes
            y_axis = np.cross(approach, fake_up)
            y_axis /= np.linalg.norm(y_axis)
            x_axis = np.cross(y_axis, approach)
            x_axis /= np.linalg.norm(x_axis)

            # Construct rotation matrix
            rot = np.eye(4)
            rot[:3, 0] = x_axis
            rot[:3, 1] = y_axis
            rot[:3, 2] = approach

            quat = quaternion_from_matrix(rot)

            pose = Pose()
            pose.position.x = midpoint[0]
            pose.position.y = midpoint[1]
            pose.position.z = midpoint[2]
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]

            return {
                "pose": pose,
                "point_1": grasp["point_i"],
                "point_2": grasp["point_j"]
            }

        return None

    def getAllGrasps(self):
        return self.allGrasps
    
    def getSelectedGrasps(self):
        return self.selectedGrasps
    