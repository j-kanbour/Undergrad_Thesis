import open3d as o3d
rom scipy.spatial import KDTree
import numpy as np
import json
import cv2

class Grasps:
    def __init__(self, superquadric, flat_plane_only = True):

        self.print = lambda *args, **kwargs: print("Grasps:", *args, **kwargs)

        self.superquadric = superquadric.getAlignedPCD()
        self.depth = cv2.imread(superquadric.getPCD().getRawData()["raw_depth"][0], cv2.IMREAD_UNCHANGED)
        self.depth_map = o3d.geometry.Image(self.depth.astype(np.uint16))
        self.camera_info = superquadric.getPCD().getRawData()["camera_info"]

        with open(self.camera_info, 'r') as f:
            scene_info = json.load(f)["0"]
            self.K = np.array(scene_info["cam_K"]).reshape(3, 3)
            self.depth_scale = float(scene_info.get("depth_scale", 1.0))

        self.object_pcd = superquadric.getPCD().getPCD()

        self.flat_plane_only = flat_plane_only

        self.allGrasps = self.generateGrasps()
        self.selectedGrasps = self.selectGrasps()


    def generateGrasps(self, num_grasps=1000, d_thresh=1.0, angle_thresh_deg=0.5):
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
        import numpy as np

        points = np.asarray(self.superquadric.points)
        normals = np.asarray(self.superquadric.normals)

        length_points = len(points)

        # angle_thresh_rad = np.deg2rad(angle_thresh_deg)
        # cos_angle_thresh = np.cos(angle_thresh_rad)

        candidate_grasps = []
        used_indices = set()

        for i in range(1,len(points), 50):
            if i in used_indices:
                continue

            point1 = points[i]
            normal1 = normals[i]

            #switch this to reverse points, as it may be possible to just match the point with its opposite in the list
            # for j in range(len(points)):
            # print(j)
            # if j in used_indices:
            #     continue

            second_index = length_points - i
            point2 = points[second_index]
            normal2 = normals[second_index]

            # 1. Distance check
            dist = np.linalg.norm(point1 - point2)
            if dist > d_thresh:
                continue

            # # 2. Antipodal check
            # dot_normals = np.dot(n_i, n_j)
            # if dot_normals >= -cos_angle_thresh:
            #     continue

            # # 3. Facing check
            # v_ij = p_j - p_i
            # if np.dot(v_ij, n_i) >= 0 or np.dot(-v_ij, n_j) >= 0:
            #     continue

            # Passed all checks → create grasp pair
            v_ij = point2 - point1
            approach_axis = v_ij / np.linalg.norm(v_ij)

            grasp_pose = {
                "index_i": i,
                "index_j": second_index,
                "point_i": point1.copy(),
                "point_i_normals": normal1.copy(),
                "point_j": point2.copy(),
                "point_j_normals": normal2.copy(),
                "approach_axis": approach_axis.copy(),
                "jaw_width": np.linalg.norm(v_ij)
            }

            candidate_grasps.append(grasp_pose)
            used_indices.update([i, second_index])  # strictly mark both as used

            if len(candidate_grasps) >= num_grasps:
                break

        for i in candidate_grasps:
            for e in i: 
                print(f'{e}:{i[e]}')
            print('\n')
        return candidate_grasps

        # #select first point
        # for point1 in range(num_grasps):
        #     p_point1 = points[point1]
        #     n_point1 = normals[point1]

        #     #find second point which matches requirements (antipodal, withing distance)
        #     for point2 in range(len(points)):
        #         p_point2 = points[point2]
        #         n_point2 = normals[point2]

        #         #make sure points are not the same
        #         if point1 == point2: continue

        #         #check distance
        #         dist = np.linalg.norm(p_point1 - p_point2)
        #         if dist > d_thresh:
        #             continue

        #         #check antipodal
        #         dot_normals = np.dot(n_i, n_j)
        #         if dot_normals >= -cos_angle_thresh:
        #             continue

        # from scipy.spatial import KDTree

        # p1_index = 123  # or however you select the surface point
        # p1 = np.asarray(self.superquadric.points)[p1_index]
        # n1 = np.asarray(self.superquadric.normals)[p1_index]
        # ray_origin = p1
        # ray_dir = -n1

        # points = np.asarray(self.superquadric.points)
        # kdtree = KDTree(points)

        # # Step along the ray and look for nearest points
        # n_steps = 100
        # step_size = 0.01  # tune depending on point cloud scale
        # radius = 0.005  # max orthogonal distance from ray to consider a hit
        # best_candidate = None

        # for i in range(1, n_steps):
        #     probe = ray_origin + i * step_size * ray_dir
        #     idxs = kdtree.query_ball_point(probe, radius)
        #     if idxs:
        #         # Optional: filter by normal alignment
        #         for idx in idxs:
        #             n2 = np.asarray(pcd.normals)[idx]
        #             if np.dot(n1, n2) < -0.9:  # nearly opposing
        #                 best_candidate = idx
        #                 break
        #         if best_candidate is not None:
        #             break

        # if best_candidate is not None:
        #     p2 = points[best_candidate]
        #     print("Opposing point found at:", p2)
        # else:
        #     print("No opposing point found within parameters.")

    def checkCollision(self, grasp, epsilon=0.005):
        """
        Checks whether the given grasp is occluded by other objects in the depth image
        using provided camera intrinsics (from cam_K).
        
        Args:
            grasp (dict): Grasp dictionary with 'point_i' and 'point_j' in camera coordinates.
            epsilon (float): Margin to account for depth noise (in metres).

        Returns:
            True if either point is occluded, False if both are visible.
        """

        # Get depth image and intrinsics
        K = self.K
        height, width = self.depth.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

        def is_occluded(p):
            X, Y, Z = p
            if Z <= 0:
                return True  # invalid depth

            # Project point to pixel
            u = int(round((X * fx) / Z + cx))
            v = int(round((Y * fy) / Z + cy))

            # Bounds check
            if not (0 <= u < width and 0 <= v < height):
                return True

            depth_at_pixel = self.depth_map[v, u]

            return Z > depth_at_pixel + epsilon  # occluded if something is closer

        # Check both grasp contact points
        print(is_occluded(grasp["point_i"]), is_occluded(grasp["point_j"]))
        return is_occluded(grasp["point_i"]) or is_occluded(grasp["point_j"])

    def horizontalPlace(self, grasp, angle_thresh_deg=10):
        """
        Determines whether the grasp approach axis is aligned with the vertical direction
        (e.g., gripper approaching from above the object, suitable for horizontal placement).

        Args:
            grasp (dict): Grasp info with an 'approach_axis' field (3D unit vector).
            angle_thresh_deg (float): Allowed deviation from the vertical axis (Z-up).

        Returns:
            True if the grasp is within the allowed angular threshold from vertical, else False.
        """
        if not self.flat_plane_only:
            return True  # Allow all grasps if not restricted

        # Get the approach vector (unit 3D vector)
        approach = grasp.get("approach_axis", None)
        if approach is None:
            return False  # No approach axis means we cannot check alignment

        # Z-up unit vector (e.g., vertical in camera or world coordinates)
        z_axis = np.array([0, 0, 1])

        # Compute angle between approach and z-axis
        dot_product = np.dot(approach, z_axis)
        angle_rad = np.arccos(np.clip(abs(dot_product), -1.0, 1.0))  # absolute to account for both +z and -z
        angle_deg = np.degrees(angle_rad)
        print(angle_deg, angle_thresh_deg)
        # Return True if the grasp is close enough to vertical
        return angle_deg <= angle_thresh_deg
    
    def crossVisibleSurface(self, grasp, threshold=0.02):
        """
        Checks if the grasp spans the visible surface: points lie on opposite sides of the object
        and the grasp line crosses through the visible region.

        Args:
            grasp (dict): Contains 'point_i' and 'point_j' in camera coordinates.
            threshold (float): Max distance for midpoint to be considered "on" the object.

        Returns:
            True if the grasp spans the visible object, False otherwise.
        """

        visible_points = np.asarray(self.object_pcd.points)
        if len(visible_points) == 0:
            return False

        tree = KDTree(visible_points)

        p1 = grasp["point_i"]
        p2 = grasp["point_j"]
        midpoint = (p1 + p2) / 2

        # Check if midpoint lies close to object surface
        midpoint_dist, _ = tree.query(midpoint)
        if midpoint_dist > threshold:
            return False

        # Fit PCA plane to visible object
        # (to estimate object "side" using surface normal direction)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        pca.fit(visible_points)
        normal = pca.components_[0]

        # Project both points onto the principal axis
        proj1 = np.dot(p1 - midpoint, normal)
        proj2 = np.dot(p2 - midpoint, normal)

        # If they project to opposite signs, they lie on opposite sides
        return proj1 * proj2 < 0

    def selectGrasps(self):
        pass
        """
            Selection priority list: 
                - no other obsticles in the way
                - points corss over the visible surface of the actual object
                - possible force closure
                - Aligned with estimated 6-DOF Pose Estimation
                - Prepare for two possibilities
                    - Grasp where the robot can not tilt
                    - Grasp where the robot can tilt its hand
                - Grasps from front plane
                - Grasps from top down
                - Provide center point for grasp
                - Grasp the rim (inside of an object) i.e. bowl, cup, ...
        """

        # self.allGrasps = self.allGrasps
        # self.superquadric = self.superquadric.getAlignedPCD()
        # self.depth_map = self.superquadric.getPCD().getRawData()["raw_depth"]
        # object_pcd = self.superquadric.getPCD()

        for grasp in self.allGrasps:
            if  self.horizontalPlace(grasp):
                return [grasp]
        
    def getAllGrasps(self):
        return self.allGrasps
    
    def getSelectedGrasps(self):
        return self.selectedGrasps