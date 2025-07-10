#!/usr/bin/env python3.8
from pointCloudData import PointCloudData
import numpy as np
import open3d as o3d
from scipy.stats import kurtosis
from typing import Tuple
import re
from scipy.optimize import least_squares

class Superquadric:
    def __init__(self, object_ID, class_name, pcd):
        self.print = lambda *args, **kwargs: print("Superquadric:", *args, **kwargs)

        self.object_ID = object_ID
        self.class_name = class_name.lower()

        #built point cloud from raw data
        self.pcdObejct = pcd
        self.pointCloudObject = pcd.getPCD()
        self.rawData = pcd.getRawData()

        #estimate values of e
        self.e1, self.e2 = self.estimateE()

        self.superquadric = self.fit_superquadric_cloud()

        #using ICP aligned the superquadric estimate to the target object
        self.aligned_PCD = self.createSuperquadricAsPCD()
                
    def estimateE(self):
        # Use multiple frames or add smoothing
        centroid = self.pcdObejct.getCentroid()
        print(centroid)
        
        # Calculate kurtosis for each axis separately
        krt_x = kurtosis(centroid[0], axis=0, fisher=True, bias=False)
        krt_y = kurtosis(centroid[1], axis=0, fisher=True, bias=False)
        krt_z = kurtosis(centroid[2], axis=0, fisher=True, bias=False)
        
        # Add bounds checking and smoothing
        krt_x = np.clip(krt_x, -2, 10)
        krt_y = np.clip(krt_y, -2, 10)
        krt_z = np.clip(krt_z, -2, 10)
        
        # Reduce sensitivity
        e1 = np.clip(1 + (krt_z - 3) * 0.05, 0.3, 2.0)  # Using z-axis kurtosis
        e2 = np.clip(1 + ((krt_x + krt_y) / 2 - 3) * 0.05, 0.3, 2.0)  # Using x,y average
        
        return e1, e2
        
    # def defineE(self):
    #     try:
    #         cn = (self.class_name or "").lower()
    #         # if re.search(r"can|cup|mug", cn):   return 0.1, 1.0   # cylinder
    #         # if re.search(r"box", cn):           return 0.3, 0.3   # cuboid
    #         # if re.search(r"ball|sphere", cn):   return 1.0, 1.0   # sphere
    #         # if re.search(r"bowl|plate", cn):    return 0.6, 0.6
    #         return self.estimateE()   
    #     except Exception as e:
    #         print(f'defineE Error: {e}')
    #         return 1, 1
    
    def pca_align(self):
        xyz = np.asarray(self.pointCloudObject.points)
        centre = xyz.mean(0)
        xyz_c = xyz - centre
        eigv, eigvec = np.linalg.eigh(np.cov(xyz_c.T))
        
        # Sort by eigenvalue magnitude
        idx = eigv.argsort()[::-1]
        eigv = eigv[idx]
        eigvec = eigvec[:, idx]
        
        # Ensure consistent eigenvector orientation
        for i in range(3):
            if eigvec[i, i] < 0:
                eigvec[:, i] *= -1
        
        # Check for near-degenerate cases
        if eigv[0] / eigv[1] < 1.1 or eigv[1] / eigv[2] < 1.1:
            # Use more stable alignment for near-spherical objects
            pass  # Consider alternative alignment
        
        return (xyz_c @ eigvec, eigvec, centre)

    # ───────────── residual (radial-weighted) ───────────────────
    def _sq_F(self, a1,a2,a3,e1,e2, xyz):
        x,y,z = xyz[:,0]/a1, xyz[:,1]/a2, xyz[:,2]/a3
        f = (np.abs(x)**(2/e2)+np.abs(y)**(2/e2))**(e2/e1) + np.abs(z)**(2/e1) - 1
        return f

    def _res_scales(self, a, xyz, e1, e2):
        return np.linalg.norm(xyz,axis=1) * self._sq_F(a[0],a[1],a[2], e1,e2, xyz)

    # ───────────── scale optimiser (ε fixed) ────────────────────
    def fit_scales(self, xyz: np.ndarray, e1: float, e2: float):
        # More robust initial guess
        a0 = np.percentile(np.abs(xyz), 90, axis=0)  # Use 90th percentile instead of max
        
        # Add bounds to prevent unrealistic scales
        bounds = (a0 * 0.1, a0 * 10)
        
        res = least_squares(self._res_scales, a0, args=(xyz, e1, e2),
                            method='trf', bounds=bounds, max_nfev=200)
        mse = res.cost / xyz.shape[0]
        return res.x, mse

    # ───────────── dense SQ sampler ─────────────────────────────
    def sample_sq(self, a1,a2,a3,e1,e2, n_th=72, n_ph=144):
        th = np.linspace(-np.pi/2, np.pi/2, n_th)
        ph = np.linspace(-np.pi,   np.pi,   n_ph)
        th,ph = np.meshgrid(th, ph, indexing='ij'); th,ph = th.ravel(), ph.ravel()
        ce = np.sign(np.cos(th))*np.abs(np.cos(th))**e1
        se = np.sign(np.sin(th))*np.abs(np.sin(th))**e1
        co = np.sign(np.cos(ph))*np.abs(np.cos(ph))**e2
        so = np.sign(np.sin(ph))*np.abs(np.sin(ph))**e2
        pts = np.column_stack((a1*ce*co, a2*ce*so, a3*se))
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

    # ───────────── public API ───────────────────────────────────
    def fit_superquadric_cloud(self):

        # 2. PCA align
        xyz_aligned, R, centre = self.pca_align()

        # 4. optimise scales only
        scales, mse = self.fit_scales(xyz_aligned, self.e1, self.e2)
        a1,a2,a3 = scales

        # 5. sample & transform back
        sq = self.sample_sq(a1,a2,a3,self.e1,self.e2)
        sq.points = o3d.utility.Vector3dVector(np.asarray(sq.points) @ R.T + centre)
        sq.estimate_normals()

        params = dict(a1=a1,a2=a2,a3=a3,e1=self.e1,e2=self.e2,rms=np.sqrt(mse))
        return sq, params 
            
    # def createSuperquadric(self):
        
    #     try:
    #         e1, e2 = self.e1, self.e2

    #         boundingBox = self.pcd.getBoundingBox()
    #         extent = boundingBox.extent
    
    #         alpha1 = extent[0] / 2
    #         alpha2 = extent[1] / 2
    #         alpha3 = extent[2] / 2

    #         def fexp(x,p):
    #             return (np.sign(x) * (np.abs(x)**p))

    #         phi, theta = np.mgrid[0:np.pi:80j, 0:2*np.pi:80j]

    #         #x,y,z formula for superquadirc
    #         x = alpha1 * (fexp(np.sin(phi),e1)) * (fexp(np.cos(theta),e2))
    #         y = alpha2 * (fexp(np.sin(phi),e1)) * (fexp(np.sin(theta),e2))
    #         z = alpha3 * (fexp(np.cos(phi),e1))
            
    #         axis = self.pcd.getAxis()  # 3x3 rotation matrix
    #         center = self.pcd.getCentroid() # 3D centre of the bounding box

    #         # Stack your generated superquadric grid into points
    #         points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T  # (N, 3)

    #         # Transform points:
    #         #   - First rotate them using the OBB axes
    #         #   - Then translate them to the OBB centre
    #         points_transformed = points @ axis.T  # (N, 3)

    #         # Unpack back to x_final, y_final, z_final in original grid shape
    #         x_final = points_transformed[:, 0].reshape(x.shape) + center[0]
    #         y_final = points_transformed[:, 1].reshape(y.shape) + center[1]
    #         z_final = points_transformed[:, 2].reshape(z.shape) + center[2]

    #         return x_final, y_final, z_final
    #     except Exception as e:
    #         print(f"[createSuperquadric] Error: {e}")
    #         return [None] * 3

    # def alignWithICP(self):
    #     """
    #     try to make this redundant by improving the above, cause this fuckign kills speed
    #     """
    #     try:
    #         s = self.rawSuperquadric
    #         t = self.pcd.getPCD()

    #         source = copy.deepcopy(s)
    #         target = copy.deepcopy(t)

    #         threshold = 0.02
    #         voxel_size = threshold / 2

    #         # Optional downsampling (improves ICP stability)
    #         source_down = source.voxel_down_sample(voxel_size)
    #         target_down = target.voxel_down_sample(voxel_size)

    #         trans_init = np.eye(4)

    #         reg_p2p = o3d.pipelines.registration.registration_icp(
    #             source_down, target_down, threshold, trans_init,
    #             o3d.pipelines.registration.TransformationEstimationPointToPoint()
    #         )
    #         self.print("ICP Fitness:", reg_p2p.fitness)
    #         self.print("ICP Inlier RMSE:", reg_p2p.inlier_rmse)

    #         # Transform model values
    #         x, y, z = self.modelValues
    #         points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    #         points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    #         transformed_points = (reg_p2p.transformation @ points_hom.T).T[:, :3]

    #         x_final = transformed_points[:, 0].reshape(x.shape)
    #         y_final = transformed_points[:, 1].reshape(y.shape)
    #         z_final = transformed_points[:, 2].reshape(z.shape)
    #         self.modelValues = (x_final, y_final, z_final)

    #         # Create final aligned point cloud
    #         aligned_pcd = o3d.geometry.PointCloud()
    #         aligned_pcd.points = o3d.utility.Vector3dVector(transformed_points)

    #         # Step 1: estimate normals (safe)
    #         aligned_pcd.estimate_normals(
    #             search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=100)
    #         )

    #         # Step 2: optionally orient normals (on downsampled points to avoid Qhull crash)
    #         try:
    #             aligned_down = aligned_pcd.voxel_down_sample(voxel_size=0.005)
    #             aligned_down.estimate_normals(
    #                 search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
    #             )
    #             aligned_down.orient_normals_consistent_tangent_plane(k=10)

    #             # Transfer normals back (approximate)
    #             from scipy.spatial import cKDTree
    #             source_points = np.asarray(aligned_down.points)
    #             source_normals = np.asarray(aligned_down.normals)
    #             full_points = np.asarray(aligned_pcd.points)

    #             tree = cKDTree(source_points)
    #             _, indices = tree.query(full_points)
    #             aligned_pcd.normals = o3d.utility.Vector3dVector(source_normals[indices])
    #         except Exception as e:
    #             self.print("Normal orientation skipped (safe fallback):", e)

    #         return aligned_pcd
        
    #     except Exception as e:
    #         print(f"[alignWithICP] Error: {e}")
    #         return None

    def createSuperquadricAsPCD(self):
        """Builds PCD based on superquadric values"""
        try:
            # points = np.vstack((self.modelValues[0].flatten(),
            #                     self.modelValues[1].flatten(),
            #                     self.modelValues[2].flatten())).T
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(points)
            
            # #need to estimate point normals

            # #NOTE 2: good shape and coverage, calculates volume aswell (too many points)
            # #[Open3D WARNING] [CreateFromPointCloudAlphaShape] invalid tetra in TetraMesh
            sq_pcd = self.superquadric[0]
            # tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(sq_pcd)
            # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            #     sq_pcd, 0.5, tetra_mesh, pt_map)
            # mesh.compute_vertex_normals()

            # sq_pcd = mesh.sample_points_uniformly(
            #         number_of_points=10000                       # pick the density you need
            #     )
            return sq_pcd

        except Exception as e:
            print(f"[createSuperquadricAsPCD] Error: {e}")
            return None

    def getSuperquadricAsPCD(self):
        return self.aligned_PCD

    def getPCD(self):
        return self.pcdObejct
    
    def getRawData(self):
        return self.rawData
    
    # def getAlignedPCD(self):
    #     return self.aligned_PCD

"""

    TODO

    estimation is sometimes good, sometimes shit but keeps shifting for still images
    therefore not consistent 

    Model Convex objects (i.e. bowl and cup)

"""


