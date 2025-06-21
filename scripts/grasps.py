import open3d as o3d
from scipy.spatial import KDTree
import numpy as np
import json
import cv2

class Grasps:
    def __init__(self, superquadric, flat_plane_only = True):

        self.print = lambda *args, **kwargs: print("Grasps:", *args, **kwargs)

        self.superquadric = superquadric.getAlignedPCD()
        self.depth = cv2.imread(superquadric.getPCD().getRawData()["raw_depth"][0], cv2.IMREAD_UNCHANGED)
        self.depth_map = o3d.geometry.Image(self.depth.astype(np.uint16))
        self.mask = cv2.imread(superquadric.getPCD().getRawData()["raw_mask"][0], cv2.IMREAD_GRAYSCALE)
        self.camera_info = superquadric.getPCD().getRawData()["camera_info"]

        with open(self.camera_info, 'r') as f:
            scene_info = json.load(f)["0"]
            self.K = np.array(scene_info["cam_K"]).reshape(3, 3)
            self.depth_scale = float(scene_info.get("depth_scale", 1.0))

        self.object_pcd = superquadric.getPCD().getPCD()

        self.flat_plane_only = flat_plane_only

        self.allGrasps = self.generateGrasps()
        self.selectedGrasps = self.selectGrasps()

    def generateGrasps(self, num_grasps=10000, d_thresh=1.0, angle_thresh_deg=0.5):
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

            normal1 = normal1 / np.linalg.norm(normal1)  # ensure it's a unit vector
            y_axis = np.array([0, 1, 0])
            dot = np.dot(normal1, y_axis)

            # Angle from XZ plane = angle between normal and Y-axis
            angle_rad = np.arcsin(np.clip(abs(dot), -1.0, 1.0))  # deviation from parallel to XZ
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
                "angle_to_xz": angle_deg  # angle between approach and XZ plane
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

        with open(self.camera_info, 'r') as f:
            scene_info = json.load(f)["0"]
            K = np.array(scene_info["cam_K"]).reshape(3, 3)
            depth_scale = float(scene_info.get("depth_scale", 1.0))
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        height, width = self.depth.shape
        depth = self.depth.astype(np.float32) / depth_scale  # Convert to metres
        mask = self.mask  # 8-bit single-channel image

        def is_occluded(p):
            X, Y, Z = p

            # Project to image space
            u = int(round((X * fx) / Z + cx))
            v = int(round((Y * fy) / Z + cy))

            if not (0 <= u < width and 0 <= v < height):
                return True  # Out of bounds

            # Create margin window (within 10 mm radius)
            pixel_radius = int(margin * fx / Z)  # convert to pixel size based on fx

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

    def horizontalPlace(self, grasp, angle_thresh_deg=10):
        if not self.flat_plane_only:
            return True  # Allow all grasps if unrestricted

        angle = grasp.get("angle_to_xz", None)
        if angle is None:
            return False  # Cannot check without angle

        # Small angle → normal is close to XZ plane (i.e., grasp is horizontal)
        return angle <= angle_thresh_deg
        
    def crossVisibleSurface(self, grasp, threshold=0.02):
        """
        Ensures that the grasp crosses the visible object surface from one side to the other,
        and that the approach is directed into the visible part of the object.

        Returns:
            True if the grasp crosses the object and is visible-side-inward.
        """

        visible_points = np.asarray(self.object_pcd.points)
        if len(visible_points) == 0:
            return False

        # Build KD-Tree for surface distance checking
        tree = KDTree(visible_points)

        p1 = grasp["point_i"]
        p2 = grasp["point_j"]
        midpoint = (p1 + p2) / 2.0

        # Check if midpoint lies near the object surface
        midpoint_dist, _ = tree.query(midpoint)
        if midpoint_dist > threshold:
            return False

        # Use PCA to estimate the principal direction (e.g., object’s left-right axis)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        pca.fit(visible_points)
        principal_axis = pca.components_[0]

        # Ensure the principal axis always points toward the camera (arbitrary)
        if principal_axis[2] > 0:
            principal_axis = -principal_axis

        # Project both grasp points onto the principal axis
        proj1 = np.dot(p1 - midpoint, principal_axis)
        proj2 = np.dot(p2 - midpoint, principal_axis)

        # Check if points lie on opposite sides (i.e., signs of projections differ)
        if proj1 * proj2 >= 0:
            return False  # not crossing the visible surface

        # Check that the approach vector is directed into the visible side
        v_ij = p2 - p1  # grasp direction
        approach_axis = v_ij / np.linalg.norm(v_ij)

        # We want grasp to enter towards the object — so approach should oppose principal axis
        dot = np.dot(approach_axis, principal_axis)
        if dot > 0:
            return False  # grasp moves away from the visible side

        return True  # all conditions met


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

    def selectGrasps(self):
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
        #return self.allGrasps
        for grasp in self.allGrasps:
            if self.checkAntipodal(grasp) and self.checkCollision(grasp) and self.crossVisibleSurface(grasp):
                print('\n',grasp,'\n')
                return [grasp]
        
    def getAllGrasps(self):
        return self.allGrasps
    
    def getSelectedGrasps(self):
        return self.selectedGrasps