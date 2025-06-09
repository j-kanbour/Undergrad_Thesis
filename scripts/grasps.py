import open3d as o3d
import numpy as np

class Grasps:
    def __init__(self, superquadric_pcd):
        self.pcd = superquadric_pcd
        self.grasps = []

    def compute_normals(self, radius=0.01, max_nn=30):
        self.pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
        )
        normals = np.asarray(self.pcd.normals)
        return normals

    def generate_antipodal_grasps(self, num_grasps=10, angle_tolerance=np.deg2rad(15)):
        points = np.asarray(self.pcd.points)
        normals = self.compute_normals()

        grasps = []
        attempts = 0
        max_attempts = num_grasps * 50  # Avoid infinite loops

        while len(grasps) < num_grasps and attempts < max_attempts:
            # Randomly sample two points
            idx1, idx2 = np.random.choice(len(points), 2, replace=False)
            p1, p2 = points[idx1], points[idx2]
            n1, n2 = normals[idx1], normals[idx2]

            # Check if normals are approximately opposite
            dot_product = np.dot(n1, -n2)
            angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

            if angle < angle_tolerance:
                # Valid antipodal grasp
                grasp_position = (p1 + p2) / 2.0
                axis = (p2 - p1)
                axis /= np.linalg.norm(axis)  # x-axis of gripper

                # Approach direction → we use the average normal (assuming symmetric contact)
                approach_axis = (n1 + n2) / 2.0
                approach_axis /= np.linalg.norm(approach_axis)  # z-axis of gripper

                # Binormal → y-axis = approach_axis cross axis
                binormal_axis = np.cross(approach_axis, axis)
                binormal_axis /= np.linalg.norm(binormal_axis)

                # Re-orthogonalise axis
                axis = np.cross(binormal_axis, approach_axis)
                axis /= np.linalg.norm(axis)

                # Final orientation matrix (columns are x, y, z axes of gripper)
                orientation = np.stack([axis, binormal_axis, approach_axis], axis=1)

                jaw_width = np.linalg.norm(p2 - p1)

                grasps.append({
                    "position": grasp_position,
                    "orientation": orientation,
                    "approach_axis": approach_axis,
                    "binormal_axis": binormal_axis,
                    "axis": axis,
                    "jaw_width": jaw_width
                })

            attempts += 1

        self.grasps = grasps
        return grasps


    def get_grasps(self):
        return self.grasps
