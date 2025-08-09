#!/usr/bin/env python3.8

import numpy as np
import open3d as o3d
from scipy.stats import kurtosis
import copy

class Superquadric:
    def __init__(self, object_ID, class_name, pcd):
        self.print = lambda *args, **kwargs: print("Superquadric:", *args, **kwargs)

        self.object_ID = object_ID
        self.class_name = class_name.lower()

        #built point cloud from raw data
        self.pcdObejct = pcd
        self.pointCloudModel = pcd.getPCD()
        self.rawData = pcd.getRawData()

        #estimate values of e
        self.e1, self.e2 = self.defineE()

        # self.sq_pcd, self.sq_params = self.fit_superquadric_cloud()

        #determine superquadric values
        self.modelValues = self.createSuperquadric()

        self.rawSuperquadric = self.createSuperquadricAsPCD()

        #using ICP aligned the superquadric estimate to the target object
        self.aligned_PCD = self.alignWithICP()

        #using ICP aligned the superquadric estimate to the target object
        # self.aligned_PCD = self.createSuperquadricAsPCD()
                
    def estimateE(self):
        # Use multiple frames or add smoothing

        #get point cloud as point array
        points = np.asarray(self.pointCloudModel.points)

        if points.shape[0] < 10:
            self.print("Not enough points to estimate shape reliably.")
            return 1.0, 1.0

        centroid = self.pcdObejct.getCentroid()
        centered_points = points - centroid

        # Compute Fisher kurtosis for x, y, z axes of the point cloud
        #   Kurtosis:  is a statistical measure that describes the "tailedness" of a 
        #              probability distribution, essentially indicating how many outliers are present
        krt = kurtosis(centered_points, axis=0, fisher=True, bias=False)

        # Shape parameter along z-axis based on kurtosis (controls superquadric elongation or flattening)
        e1 = np.clip(1 + (krt[2] - 3) * 0.1, 0.3, 2.0)

        # Shape parameter along xy-plane based on average x and y kurtosis
        e2 = np.clip(1 + ((krt[0] + krt[1]) / 2 - 3) * 0.1, 0.3, 2.0)

        return e1, e2
        
    def defineE(self):
        try:
            # cn = (self.class_name or "").lower()
            # if re.search(r"can|cup|mug", cn):   return 0.1, 1.0   # cylinder
            # if re.search(r"box", cn):           return 0.3, 0.3   # cuboid
            # if re.search(r"ball|sphere", cn):   return 1.0, 1.0   # sphere
            # if re.search(r"bowl|plate", cn):    return 0.6, 0.6
            return self.estimateE()   
        except Exception as e:
            print(f'defineE Error: {e}')
            return 1, 1
    
    # def pca_align(self):
    #     xyz = np.asarray(self.pointCloudModel.points)
    #     center = xyz.mean(0)
    #     xyz_c = xyz - center
    #     eigv, eigvec = np.linalg.eigh(np.cov(xyz_c.T))
        
    #     # Sort by eigenvalue magnitude
    #     idx = eigv.argsort()[::-1]
    #     eigv = eigv[idx]
    #     eigvec = eigvec[:, idx]
        
    #     # Ensure consistent eigenvector orientation
    #     for i in range(3):
    #         if eigvec[i, i] < 0:
    #             eigvec[:, i] *= -1
        
    #     # Check for near-degenerate cases
    #     if eigv[0] / eigv[1] < 1.1 or eigv[1] / eigv[2] < 1.1:
    #         # Use more stable alignment for near-spherical objects
    #         pass  # Consider alternative alignment
        
    #     return (xyz_c @ eigvec, eigvec, center)

    # # ───────────── residual (radial-weighted) ───────────────────
    # def _sq_F(self, a1,a2,a3,e1,e2, xyz):
    #     x,y,z = xyz[:,0]/a1, xyz[:,1]/a2, xyz[:,2]/a3
    #     f = (np.abs(x)**(2/e2)+np.abs(y)**(2/e2))**(e2/e1) + np.abs(z)**(2/e1) - 1
    #     return f

    # def _res_scales(self, a, xyz, e1, e2):
    #     return np.linalg.norm(xyz,axis=1) * self._sq_F(a[0],a[1],a[2], e1,e2, xyz)

    # # ───────────── scale optimiser (ε fixed) ────────────────────
    # def fit_scales(self, xyz: np.ndarray, e1: float, e2: float):
    #     # More robust initial guess
    #     a0 = np.percentile(np.abs(xyz), 90, axis=0)  # Use 90th percentile instead of max
        
    #     # Add bounds to prevent unrealistic scales
    #     bounds = (a0 * 0.1, a0 * 10)
        
    #     res = least_squares(self._res_scales, a0, args=(xyz, e1, e2),
    #                         method='trf', bounds=bounds, max_nfev=200)
    #     mse = res.cost / xyz.shape[0]
    #     return res.x, mse

    # # ───────────── dense SQ sampler ─────────────────────────────
    # def sample_sq(self, a1,a2,a3,e1,e2, n_th=72, n_ph=144):
    #     th = np.linspace(-np.pi/2, np.pi/2, n_th)
    #     ph = np.linspace(-np.pi,   np.pi,   n_ph)
    #     th,ph = np.meshgrid(th, ph, indexing='ij'); th,ph = th.ravel(), ph.ravel()
    #     ce = np.sign(np.cos(th))*np.abs(np.cos(th))**e1
    #     se = np.sign(np.sin(th))*np.abs(np.sin(th))**e1
    #     co = np.sign(np.cos(ph))*np.abs(np.cos(ph))**e2
    #     so = np.sign(np.sin(ph))*np.abs(np.sin(ph))**e2
    #     pts = np.column_stack((a1*ce*co, a2*ce*so, a3*se))
    #     return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

    # # ───────────── public API ───────────────────────────────────
    # def fit_superquadric_cloud(self):

    #     # 2. PCA align
    #     xyz_aligned, R, center = self.pca_align()

    #     # 4. optimise scales only
    #     scales, mse = self.fit_scales(xyz_aligned, self.e1, self.e2)
    #     a1,a2,a3 = scales

    #     # 5. sample & transform back
    #     sq = self.sample_sq(a1,a2,a3,self.e1,self.e2)
    #     sq.points = o3d.utility.Vector3dVector(np.asarray(sq.points) @ R.T + center)
    #     sq.estimate_normals()

    #     params = dict(a1=a1,a2=a2,a3=a3,e1=self.e1,e2=self.e2,rms=np.sqrt(mse))
    #     sq = self.make_concave_if_bowl_or_cup(sq)
    #     return sq, params 
            
    def createSuperquadric(self):

        e1, e2 = self.e1, self.e2

        boundingBox = self.pcdObejct.getBoundingBox()
        extent = boundingBox.extent
 
        alpha1 = extent[0] / 2
        alpha2 = extent[1] / 2
        alpha3 = extent[2] / 2

        def fexp(x,p):
            return (np.sign(x) * (np.abs(x)**p))

        phi, theta = np.mgrid[0:np.pi:80j, 0:2*np.pi:80j]

        x = alpha1 * (fexp(np.sin(phi),e1)) * (fexp(np.cos(theta),e2))
        y = alpha2 * (fexp(np.sin(phi),e1)) * (fexp(np.sin(theta),e2))
        z = alpha3 * (fexp(np.cos(phi),e1))
        
        axis = self.pcdObejct.getAxis()  # 3x3 rotation matrix
        center = self.pcdObejct.getCentroid() # 3D centre of the bounding box

        # Stack your generated superquadric grid into points
        points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T  # (N, 3)

        # Transform points:
        #   - First rotate them using the OBB axes
        #   - Then translate them to the OBB centre
        points_transformed = points @ axis.T  # (N, 3)

        # Unpack back to x_final, y_final, z_final in original grid shape
        x_final = points_transformed[:, 0].reshape(x.shape) + center[0]
        y_final = points_transformed[:, 1].reshape(y.shape) + center[1]
        z_final = points_transformed[:, 2].reshape(z.shape) + center[2]

        return x_final, y_final, z_final

    def alignWithICP(self):
        """
        Aligns the raw superquadric to the visible point cloud using Point-to-Point ICP,
        transforms the parametric mesh points, and estimates surface normals.
        """ 
        s = self.rawSuperquadric
        t = self.pointCloudModel

        source = copy.deepcopy(s)
        target = copy.deepcopy(t)

        threshold = 0.02
        voxel_size = threshold / 2

        # Optional downsampling (improves ICP stability)
        source_down = source.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)

        trans_init = np.eye(4)

        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_down, target_down, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        # Transform model values
        x, y, z = self.modelValues
        points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
        points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
        transformed_points = (reg_p2p.transformation @ points_hom.T).T[:, :3]

        x_final = transformed_points[:, 0].reshape(x.shape)
        y_final = transformed_points[:, 1].reshape(y.shape)
        z_final = transformed_points[:, 2].reshape(z.shape)
        self.modelValues = (x_final, y_final, z_final)

        # Create final aligned point cloud
        aligned_pcd = o3d.geometry.PointCloud()
        aligned_pcd.points = o3d.utility.Vector3dVector(transformed_points)

        # Step 1: estimate normals (safe)
        aligned_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=100)
        )

        # Step 2: optionally orient normals (on downsampled points to avoid Qhull crash)
        try:
            aligned_down = aligned_pcd.voxel_down_sample(voxel_size=0.005)
            aligned_down.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
            )
            aligned_down.orient_normals_consistent_tangent_plane(k=10)

            # Transfer normals back (approximate)
            from scipy.spatial import cKDTree
            source_points = np.asarray(aligned_down.points)
            source_normals = np.asarray(aligned_down.normals)
            full_points = np.asarray(aligned_pcd.points)

            tree = cKDTree(source_points)
            _, indices = tree.query(full_points)
            aligned_pcd.normals = o3d.utility.Vector3dVector(source_normals[indices])
        except Exception as e:
            self.print("Normal orientation skipped (safe fallback):", e)
        aligned_pcd = self.make_concave_if_bowl_or_cup(aligned_pcd)
        return aligned_pcd


    def createSuperquadricAsPCD(self):
        """Builds PCD based on superquadric values"""
        points = np.vstack((self.modelValues[0].flatten(),
                            self.modelValues[1].flatten(),
                            self.modelValues[2].flatten())).T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        return pcd
        
    def make_concave_if_bowl_or_cup(self, sq_pcd):
        if "bowl" not in self.class_name:
            return  sq_pcd# Do nothing for other objects


        original = sq_pcd
        original_points = np.asarray(original.points)

        # Step 1: Shift the points upward by 5 mm (0.005 m)
        shifted_points = original_points.copy()
        shifted_points[:, 2] += 0.005

        shifted_pcd = o3d.geometry.PointCloud()
        shifted_pcd.points = o3d.utility.Vector3dVector(shifted_points)

        # Step 2: Use distance filtering to keep only points in original but not in shifted
        original_tree = o3d.geometry.KDTreeFlann(original)

        concave_points = []
        for pt in shifted_points:
            [k, idx, _] = original_tree.search_radius_vector_3d(pt, 0.002)
            if k == 0:
                concave_points.append(pt)

        # Combine original and subtractive to form a concave shell
        concave_points = np.vstack(concave_points) if concave_points else original_points

        concave_pcd = o3d.geometry.PointCloud()
        concave_pcd.points = o3d.utility.Vector3dVector(concave_points)
        concave_pcd.estimate_normals()

        return concave_pcd


    def getSuperquadricAsPCD(self):
        if self.aligned_PCD: 
            return self.aligned_PCD
        return self.createSuperquadricAsPCD()
    
    def getSuperquadricParams(self):
        return self.sq_params

    def getPCD(self):
        return self.pcdObejct
    
    def getRawData(self):
        return self.rawData
    
    # def getAlignedPCD(self):
    #     return self.aligned_PCD


