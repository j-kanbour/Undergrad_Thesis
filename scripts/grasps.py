import open3d as o3d
import numpy as np

class Grasps:
    def __init__(self, superquadric):

        self.print = lambda *args, **kwargs: print("Grasps:", *args, **kwargs)

        self.superquadric = superquadric.getAlignedPCD()
        self.depth_map = superquadric.getPCD().getRawData()["raw_depth"]

        self.allGrasps = self.generateGrasps()
        self.selectedGrasps = self.selectGrasps()

    def generateGrasps(self, num_grasps=6399, d_thresh=1.0, angle_thresh_deg=0.5):
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
            grasp_pose = {
                "index_i": i,
                "index_j": second_index,
                "point_i": point1.copy(),
                "point_i_normals": normal1.copy(),
                "point_j": point2.copy(),
                "point_j_normals": normal2.copy(),
                # "jaw_width": dist,
                # "dot_normals": dot_normals,
                # "position_check": (np.dot(v_ij, n_i), np.dot(-v_ij, n_j)),
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


    def selectGrasps(self):
        """
            Loop through all possible grasps

            Selection priority list: 
                - no other obsticles in the way
                - points corss over the visible surface of the actual object
                - possible force closure
                - Aligned with estimated 6-DOF Pose Estimation
                - Additional refinement includes checking local curvature
        """
        pass

    def getAllGrasps(self):
        return self.allGrasps
    
    def getSelectedGrasps(self):
        return self.selectedGrasps