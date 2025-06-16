from pointCloudData import PointCloudData
import numpy as np
import open3d as o3d
from scipy.stats import kurtosis
import copy
import math
import re

class Superquadric:
    def __init__(self, object_ID, class_name, input_type, raw_data_1, raw_depth=None, raw_mask=None, camera_info=None):
        self.print = lambda *args, **kwargs: print("Superquadric:", *args, **kwargs)

        self.object_ID = object_ID
        self.class_name = class_name.lower()

        #built point cloud from raw data
        self.pcd = PointCloudData(object_ID, input_type, raw_data_1, raw_depth, raw_mask, camera_info)

        #estimate values of e
        self.e1, self.e2 = self.defineE()

        #determine superquadric values
        self.modelValues = self.createSuperquadric()

        self.rawSuperquadric = self.createSuperquadricAsPCD()

        #using ICP aligned the superquadric estimate to the target object
        self.aligned_PCD = self.alignWithICP()
            
    def estimateE(self):

        """
            TODO: Check the accuracy of this method, is there any better ones in terms of accuracy and speed
            TODO: Incorporate bounds or strict numbers for certain object classes (box, can, ball, ...)
        """

        #get point cloud as point array
        points = np.asarray(self.pcd.getPCD().points)

        if points.shape[0] < 10:
            self.print("Not enough points to estimate shape reliably.")
            return 1.0, 1.0

        centroid = self.pcd.getCentroid()
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
    
    def defineE(self):
        """
            e1 and e2 bounds for different primitives

            Cylinder: 0.1, 1
            Cuboid: 0.1, 2
            Sphere: 1, 1
            ...
        """
        return self.estimateE()
        if re.search(r"can", self.class_name):
            return 0.1, 1

        elif re.search(r"box", self.class_name):
            return 0.1, 2

        elif re.search(r"ball", self.class_name):
            return 1, 1
        elif re.search(r"bowl", self.class_name):
            return 0, 0
        elif re.search(r"plate", self.class_name):
            return 0, 0
        else:
            return self.estimateE()
        

    def createSuperquadric(self):

        e1, e2 = self.e1, self.e2

        boundingBox = self.pcd.getBoundingBox()
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
        
        axis = self.pcd.getAxis()  # 3x3 rotation matrix
        center = self.pcd.getCentroid() # 3D centre of the bounding box

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
        
        """NOTE: WTF is going on here"""

        s = self.rawSuperquadric
        t = self.pcd.getPCD()

        # Safe deep copies
        source = copy.deepcopy(s)
        target = copy.deepcopy(t)

        threshold=0.02

        # Optional: downsampling (safe, improves stability)
        voxel_size = threshold / 2
        source_down = source.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)

        trans_init = np.eye(4)

        # Use PointToPoint ICP — much safer for parametric model
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_down, target_down, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        self.print("ICP Fitness:", reg_p2p.fitness)
        self.print("ICP Inlier RMSE:", reg_p2p.inlier_rmse)

        # Apply transformation to your modelValues
        x, y, z = self.modelValues
        points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
        points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
        transformed_points = (reg_p2p.transformation @ points_hom.T).T[:, :3]

        # Reshape back to original shapes
        x_final = transformed_points[:, 0].reshape(x.shape)
        y_final = transformed_points[:, 1].reshape(y.shape)
        z_final = transformed_points[:, 2].reshape(z.shape)

        self.modelValues = (x_final, y_final, z_final)

        # Return aligned superquadric PCD
        aligned_pcd = o3d.geometry.PointCloud()
        aligned_pcd.points = o3d.utility.Vector3dVector(transformed_points)
        aligned_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=100))
        aligned_pcd.orient_normals_consistent_tangent_plane(k=10)

        return aligned_pcd

    def createSuperquadricAsPCD(self):
        """Builds PCD based on superquadric values"""
        points = np.vstack((self.modelValues[0].flatten(),
                            self.modelValues[1].flatten(),
                            self.modelValues[2].flatten())).T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd
    
    def getSuperquadricAsPCD(self):
        return self.rawSuperquadric

    def getPCD(self):
        return self.pcd
    
    def getAlignedPCD(self):
        return self.aligned_PCD

    def updateSuperquadric(self):
        pass


