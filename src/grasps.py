import open3d as o3d
from scipy.spatial import cKDTree
import numpy as np
import json
import cv2
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_matrix


class Grasps:
    def __init__(self, superquadric, orientation=None):

        self.print = lambda *args, **kwargs: print("Grasps:", *args, **kwargs)

        self.superquadric = superquadric.getAlignedPCD()
        self.depth = superquadric.getPCD().getRawData()["raw_depth"]
        self.depth_map = o3d.geometry.Image(self.depth.astype(np.uint16))
        self.mask = cv2.imread(superquadric.getPCD().getRawData()["raw_mask"], cv2.IMREAD_GRAYSCALE)
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

        self.orientation = orientation
        self.allGrasps = self.generateGrasps()
        self.selectedGrasps = self.selectGrasps()

    def generateGrasps(self, num_grasps=10000, d_thresh=1.0):
        """
        Generates a list of antipodal grasp point pairs from a superquadric surface point cloud.

        Ensures a 1:1 relationship between points: each point is included in at most one grasp pair.

        Parameters:
            num_grasps (int): Number of grasp pairs to return.
            d_thresh (float): Maximum allowed Euclidean distance between point pairs.
            angle_thresh_deg (float): Minimum required angle between normals (in degrees) to qualify as antipodal.

        Returns:
            List[Dict]: A list of grasp pairs with geometric information.

        NOTE: anti podal points may not be the most advantagious, look for points that exist opposite eachother
        NOTE 2: when working with superquadrics antipodal, opposing point poits are are opposing
                ends of the point array
        """

        points = np.asarray(self.superquadric.points)
        normals = np.asarray(self.superquadric.normals)

        length_points = len(points)

        candidate_grasps = []
        used_indices = set()

        for i in range(1, len(points), 50):
            if i in used_indices:
                continue

            point1 = points[i]
            normal1 = normals[i]

            second_index = length_points - i
            point2 = points[second_index]
            normal2 = normals[second_index]

            # Distance check
            dist = np.linalg.norm(point1 - point2)
            if dist > d_thresh:
                continue

            # Vector from point1 to point2
            vec = point2 - point1
            vec_norm = vec / np.linalg.norm(vec)

            # Project vec onto the XZ plane by zeroing the Y component
            vec_xz = vec.copy()
            vec_xz[1] = 0

            # Normalize the projection
            vec_xz_norm = vec_xz / np.linalg.norm(vec_xz)

            # Compute angle between vec and its XZ projection
            dot = np.dot(vec_norm, vec_xz_norm)
            dot = np.clip(dot, -1.0, 1.0)  # Ensure within domain of arccos
            angle_rad = np.arccos(dot)
            angle_deg = np.degrees(angle_rad)

            # Create grasp pose
            grasp_pose = {
                "index_i": i,
                "index_j": second_index,
                "point_i": point1.copy(),
                "point_i_normals": normal1.copy(),
                "point_j": point2.copy(),
                "point_j_normals": normal2.copy(),
                "jaw_width": dist,
                "angle_to_xz": angle_deg  # angle between the line and the XZ plane
            }


            candidate_grasps.append(grasp_pose)
            used_indices.update([i, second_index])

            if len(candidate_grasps) >= num_grasps:
                break

        for i in candidate_grasps:
            for e in i: 
                print(f'{e}:{i[e]}')
            print('\n')

        return candidate_grasps
    
    def checkCollision(self, grasp, margin=0.1):
        """
        Checks whether the given grasp will collide with surrounding objects
        by analysing depth map and bitmask.

        Returns:
            True if both grasp points are collision-free, False otherwise.
        """
        # ===== Get camera intrinsics =====

        height, width = self.depth.shape
        depth = self.depth.astype(np.float32) / self.depth_scale  # Convert to metres
        mask = self.mask  # 8-bit single-channel image

        def is_occluded(p):
            X, Y, Z = p

            # Project to image space
            u = int(round((X * self.fx) / Z + self.cx))
            v = int(round((Y * self.fy) / Z + self.cy))

            if not (0 <= u < width and 0 <= v < height):
                return True  # Out of bounds

            # Create margin window (within 10 mm radius)
            pixel_radius = int(margin * self.fx / Z)  # convert to pixel size based on fx

            # Extract ROI around (u, v)
            u_min = max(0, u - pixel_radius)
            u_max = min(width, u + pixel_radius + 1)
            v_min = max(0, v - pixel_radius)
            v_max = min(height, v + pixel_radius + 1)

            for y in range(v_min, v_max):
                for x in range(u_min, u_max):
                    if mask[y, x] == 0:  # not part of target object
                        d = depth[y, x]
                        if 0 < d < Z - margin:  # object in front of grasp point
                            return True
            return False

        # Check both points
        occluded_i = is_occluded(grasp["point_i"])
        occluded_j = is_occluded(grasp["point_j"])

        print(occluded_i, occluded_j)

        return not (occluded_i or occluded_j)

    def horizontalPlace(self, grasp, angle_thresh_deg=1):

        #NOTE IS it right to have this relative to the image, 

        if not self.flat_plane_only:
            return True  # Allow all grasps if unrestricted

        angle = grasp.get("angle_to_xz", None)
        if angle is None:
            return False  # Cannot check without angle

        # Small angle → normal is close to XZ plane (i.e., grasp is horizontal)
        return angle <= angle_thresh_deg
        
    def checkDepth(self, grasp, depth=0.02):
        """
        Ensures that the grasp is a maximum of 'depth' deep into the side of the superquadric.

        Parameters:
            grasp: dict containing grasp information (see docstring above)
            depth: maximum allowed depth into the object

        Returns:
            True if the grasp is <= depth from the surface, False otherwise.
        """
        print(np.min(np.asarray(self.superquadric.points), axis=0))
        print(np.max(np.asarray(self.superquadric.points), axis=0))

        # Midpoint of the grasp
        midpoint = (grasp["point_i"] + grasp["point_j"]) / 2.0

        # Build a KDTree from the superquadric point cloud if not already built
        if not hasattr(self, "_sq_kdtree"):
            points = np.asarray(self.superquadric.points)
            self._sq_kdtree = cKDTree(points)

        # Query nearest surface point
        dist, _ = self._sq_kdtree.query(midpoint)

        # Check if grasp is shallow enough
        return dist <= depth

    def checkAntipodal(self, grasp, normal_dot_thresh=-0.0, alignment_thresh=0.0):
        # Extract point and normal data
        p_i = grasp['point_i']
        n_i = grasp['point_i_normals']
        p_j = grasp['point_j']
        n_j = grasp['point_j_normals']

        # Normal opposition: dot product should be strongly negative (near -1)
        dot_normals = np.dot(n_i, n_j)
        if dot_normals > normal_dot_thresh:
            return False

        # Vector from i to j
        v_ij = p_j - p_i
        v_ij /= np.linalg.norm(v_ij)

        # Alignment: normal at i should point roughly toward j (opposite v_ij)
        if np.dot(n_i, v_ij) >= alignment_thresh:
            return False

        # Normal at j should point roughly toward i (along v_ij)
        if np.dot(n_j, -v_ij) >= alignment_thresh:
            return False

        return True

    # def selectGrasps(self):
    #     """
    #         Selection priority list: 
    #             - no other obsticles in the way
    #             - points corss over the visible surface of the actual object
    #             - possible force closure
    #             - Aligned with estimated 6-DOF Pose Estimation
    #             - Prepare for two possibilities
    #                 - Grasp where the robot can not tilt
    #                 - Grasp where the robot can tilt its hand
    #             - Grasps from front plane
    #             - Grasps from top down
    #             - Provide center point for grasp
    #             - Grasp the rim (inside of an object) i.e. bowl, cup, ...
    #     """

    #     # self.allGrasps = self.allGrasps
    #     # self.superquadric = self.superquadric.getAlignedPCD()
    #     # self.depth_map = self.superquadric.getPCD().getRawData()["raw_depth"]
    #     # object_pcd = self.superquadric.getPCD()
    #     #return self.allGrasps
    #     for grasp in self.allGrasps:
    #         if self.checkAntipodal(grasp) and self.checkDepth(grasp):
    #             print('\n',grasp,'\n')
    #             return [grasp]
    def selectGrasps(self):
        """
        Selects the best grasp and returns it as a geometry_msgs/Pose.
        """
        for grasp in self.allGrasps:
            if self.checkAntipodal(grasp) and self.checkDepth(grasp) and self.horizontalPlace(grasp):
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

                return pose

        return None

    def getAllGrasps(self):
        return self.allGrasps
    
    def getSelectedGrasps(self):
        return self.selectedGrasps
    

"""

    TODO
    
    Grasp selecter needs to select a grasp which:

    - is antipodal grasps (opposing normals)
    - is not obstructed by another object
    - Grasp should be towards the edge of the object (D mm form the border of the object), depending on the depth of the gripper

    - Grasping from the side:
        - The grasp crosses over the visible side of the object
        - If a horozontal grasp is needed: the grasp needs to be parallel to the ground 
        - If a non horozontal grasp is neede: angled grasps are possible

    - Grasping from the top:
        - The grasp crosses over the visible top of the object
        - If the object is convave: need to grip from the rim
        - If the object is convex: need to grip the outside 


"""