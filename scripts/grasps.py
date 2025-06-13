import open3d as o3d
import numpy as np

class Grasps:
    def __init__(self, aligned_superquadric):

        self.print = lambda *args, **kwargs: print("Grasps:", *args, **kwargs)

        self.superquadric = aligned_superquadric
        # self.aligned_superquadric = superquadric.getAlignedPCD()

        self.grasps = self.generateGrasps()

    def generateGrasps(self, num_grasps=10, d_thresh=1, angle_thresh_deg=10):

        points = np.asarray(self.superquadric.points)
        normals = np.asarray(self.superquadric.normals)

        angle_thresh_rad = np.deg2rad(angle_thresh_deg)
        cos_angle_thresh = np.cos(angle_thresh_rad)

        candidate_grasps_pairs = []

        for i in range(len(points)):
            p_i = points[i]
            n_i = normals[i]

            for j in range(i + 1, len(points)):
                p_j = points[j]
                n_j = normals[j]

                # Check distance
                dist = np.linalg.norm(p_i - p_j)
                if dist > d_thresh:
                    continue

                # Antipodal tests
                dot_normals = np.dot(n_i, n_j)
                if dot_normals > -cos_angle_thresh:
                    continue

                v_ij = p_j - p_i
                if np.dot(v_ij, n_i) >= 0 or np.dot(-v_ij, n_j) >= 0:
                    continue

                # Build grasp
                grasp_position = (p_i + p_j) / 2.0

                x_axis = v_ij / np.linalg.norm(v_ij)

                approach = (-n_i + n_j)
                norm = np.linalg.norm(approach)
                if norm < 1e-6:
                    approach = -n_i
                else:
                    approach /= norm

                binormal_axis = np.cross(approach, x_axis)
                binormal_axis /= np.linalg.norm(binormal_axis)

                orientation = np.column_stack((x_axis, binormal_axis, approach))

                grasp_pose = {
                    "position": grasp_position,
                    "approach_axis": approach,
                    "binormal_axis": binormal_axis,
                    "axis": x_axis,
                    "jaw_width": dist,
                    "orientation": orientation,
                    "dot_normals": dot_normals,
                    "position_check": (np.dot(v_ij, n_i), np.dot(-v_ij, n_j)),
                    "point_i": p_i.copy(),
                    "point_i_normals": n_i.copy(),
                    "point_j": p_j.copy(),
                    "point_j_normals": n_j.copy(),
                }

                candidate_grasps_pairs.append(grasp_pose)

                if len(candidate_grasps_pairs) >= num_grasps:
                    return candidate_grasps_pairs
                break


        return candidate_grasps_pairs

    def selectGrasps(self):
        pass

    def getAllGrasps(self):
        return self.grasps
    
    def getSelectedGrasps(self):
        pass
