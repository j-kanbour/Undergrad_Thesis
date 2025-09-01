from pointCloudData import PointCloudData
import numpy as np
import open3d as o3d
from scipy.stats import kurtosis
import copy
import math
import re

class Superquadric:
    def __init__(self, object_ID, class_name, raw_rgb, raw_depth=None, mask=None, camera_info=None):
        self.print = lambda *args, **kwargs: print("Superquadric:", *args, **kwargs)

        self.object_ID = object_ID
        self.class_name = class_name.lower()

        #built point cloud from raw data
        self.pcd = PointCloudData(object_ID, raw_rgb, raw_depth, mask, camera_info)

        #estimate values of e
        self.e1, self.e2 = self.estimateE(class_name, self.pcd)

        self.superquadric = self.createSuperquadric(self.pcd, self.e1, self.e2,)

        #using ICP aligned the superquadric estimate to the target object
        # self.aligned_PCD = self.alignWithICP()

    def estimateE(self, class_name, pcd):
        """
            e1 and e2 bounds for different primitives

            Cylinder: 0.1, 1
            Cuboid: 0.1, 2
            Sphere: 1, 1
            ...
        """

        if re.search(r"can", class_name):
            return 0.1, 1

        elif re.search(r"box", class_name):
            return 0.1, 2

        elif re.search(r"ball", class_name):
            return 1, 1
        elif re.search(r"bowl", class_name):
            return 0, 0
        elif re.search(r"plate", class_name):
            return 0, 0
        else:
            #get point cloud as point array
            print(pcd)
            points = np.asarray(pcd.getPCD().points)

            if points.shape[0] < 10:
                self.print("Not enough points to estimate shape reliably.")
                return 1.0, 1.0

            centroid = pcd.getCentroid()
            centered_points = points - centroid

            # Compute Fisher kurtosis for x, y, z axes of the point cloud
            #   Kurtosis:  is a statistical measure that describes the "tailedness" of a 
            #              probability distribution, essentially indicating how many outliers are present
            krt = kurtosis(centered_points, axis=0, fisher=True, bias=False)

            # Shape parameter along z-axis based on kurtosis (controls superquadric elongation or flattening)
            e1 = np.clip(1 + (krt[2] - 3) * 0.1, 0.3, 2.0)

            # Shape parameter along xy-plane based on average x and y kurtosis
            e2 = np.clip(1 + ((krt[0] + krt[1]) / 2 - 3) * 0.1, 0.3, 2.0)

            self.print(f"Estimated e1: {e1:.3f}, e2: {e2:.3f}")
            return e1, e2
            
    def createSuperquadric(self, pcd, e1, e2, res_u, res_v, n_points=40000):
        
        eta = np.linspace(-np.pi/2, np.pi/2, res_u)
        omega = np.linspace(-np.pi, np.pi, res_v, endpoint=False)
        Eta, Omega = np.meshgrid(eta, omega, indexing="ij")

        extentOfPcd = pcd.getBoundingBox().extent
        a1, a2, a3 = extentOfPcd[0]/2, extentOfPcd[1]/2, extentOfPcd[2]/2, 

        def sgn(x):  # sign with zero preserved
            return np.sign(x + 1e-15)

        ce, se = np.cos(Eta), np.sin(Eta)
        co, so = np.cos(Omega), np.sin(Omega)

        x = a1 * sgn(ce) * np.abs(ce)**e1 * sgn(co) * np.abs(co)**e2
        y = a2 * sgn(ce) * np.abs(ce)**e1 * sgn(so) * np.abs(so)**e2
        z = a3 * sgn(se) * np.abs(se)**e1

        V = np.stack([x, y, z], axis=-1).reshape(-1, 3)

        faces = []
        for i in range(res_u - 1):
            for j in range(res_v):
                jn = (j + 1) % res_v
                v00 = i * res_v + j
                v01 = i * res_v + jn
                v10 = (i + 1) * res_v + j
                v11 = (i + 1) * res_v + jn
                faces.append([v00, v10, v11])
                faces.append([v00, v11, v01])

        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(V),
            triangles=o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32)),
        )
        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()
        mesh.compute_vertex_normals()

        superquadricPcd = mesh.sample_points_poisson_disk(number_of_points=n_points, init_factor=5)
        # If you prefer a faster even triangle based sampler:
        # pcd = mesh.sample_points_uniformly(number_of_points=n_points)
        superquadricPcd.estimate_normals()
        return superquadricPcd

    # def alignWithICP(self):
        
    #     """NOTE: WTF is going on here"""

    #     s = self.superquadric
    #     t = self.pcd.getPCD()

    #     # Safe deep copies
    #     source = copy.deepcopy(s)
    #     target = copy.deepcopy(t)

    #     threshold=0.02

    #     # Optional: downsampling (safe, improves stability)
    #     voxel_size = threshold / 2
    #     source_down = source.voxel_down_sample(voxel_size)
    #     target_down = target.voxel_down_sample(voxel_size)

    #     trans_init = np.eye(4)

    #     # Use PointToPoint ICP — much safer for parametric model
    #     reg_p2p = o3d.pipelines.registration.registration_icp(
    #         source_down, target_down, threshold, trans_init,
    #         o3d.pipelines.registration.TransformationEstimationPointToPoint()
    #     )
    #     self.print("ICP Fitness:", reg_p2p.fitness)
    #     self.print("ICP Inlier RMSE:", reg_p2p.inlier_rmse)

    #     # Apply transformation to your modelValues
    #     x, y, z = self.modelValues
    #     points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    #     points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    #     transformed_points = (reg_p2p.transformation @ points_hom.T).T[:, :3]

    #     # Reshape back to original shapes
    #     x_final = transformed_points[:, 0].reshape(x.shape)
    #     y_final = transformed_points[:, 1].reshape(y.shape)
    #     z_final = transformed_points[:, 2].reshape(z.shape)

    #     self.modelValues = (x_final, y_final, z_final)

    #     # Return aligned superquadric PCD
    #     aligned_pcd = o3d.geometry.PointCloud()
    #     aligned_pcd.points = o3d.utility.Vector3dVector(transformed_points)
    #     aligned_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=100))
    #     aligned_pcd.orient_normals_consistent_tangent_plane(k=10)

    #     return aligned_pcd
    
    def getSuperquadricAsPCD(self):
        return self.superquadric

    def getPCD(self):
        return self.pcd
    
    # def getAlignedPCD(self):
    #     return self.aligned_PCD

    def updateSuperquadric(self):
        pass