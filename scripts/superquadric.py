
from pointCloudData import PointCloudData
import numpy as np
import open3d as o3d
from scipy.stats import kurtosis

#can you work to build the superquadric directly from the RGB-D image and mask removing the need for a point cloud object

class Superquadric:
    def __init__(self, object_ID, class_name, input_type, raw_data_1, raw_depth=None, raw_mask=None, camera_info=None):
        self.print = lambda *args, **kwargs: print("Superquadric:", *args, **kwargs)
        self.object_ID = object_ID
        self.class_name = class_name
        
        self.pcd = PointCloudData(object_ID, input_type, raw_data_1, raw_depth, raw_mask, camera_info)

        self.raw_data_1 = raw_data_1[0] #rgb raw, rgbd or pcd
        self.raw_depth = raw_depth[0]
        self.raw_mask = raw_mask[0]
        self.camera_info = camera_info

        self.e1, self.e2 = self.estimateE()
        self.modelValues = self.createSuperquadric()

    def estimateE(self):
        """
        Estimate superquadric shape parameters e1 and e2 from the point cloud
        using kurtosis of the point distribution.
        """
        # Get Nx3 array of point coordinates
        points = np.asarray(self.pcd.getPCD().points)

        if points.shape[0] < 10:
            self.print("Not enough points to estimate shape reliably.")
            return 1.0, 1.0  # fallback to sphere

        # Centre the points around the centroid
        centroid = self.pcd.getCentroid()
        centered_points = points - centroid

        # Calculate kurtosis along each axis
        krt = kurtosis(centered_points, axis=0, fisher=True, bias=False)

        # Map kurtosis to shape parameters
        # Higher kurtosis → sharper (larger e), lower kurtosis → boxy (smaller e)
        e1 = np.clip(1 + (krt[2] - 3) * 0.1, 0.3, 2.0)  # vertical roundness (z-axis)
        e2 = np.clip(1 + ((krt[0] + krt[1]) / 2 - 3) * 0.1, 0.3, 2.0)  # horizontal roundness (x-y average)

        self.print(f"Estimated e1: {e1:.3f}, e2: {e2:.3f}")
        return e1, e2

    
    def createSuperquadric(self):
        """
        Generates a superquadric aligned to the principal axis.
        Scaling is applied after rotation to better fit the rotated bounding box.
        """
        # Get centroid
        c1, c2, c3 = self.pcd.getCentroid()

        # Shape parameters (adjust if needed)
        e1 = self.e1
        e2 = self.e2

        # Angular resolution
        n_theta = 100
        n_phi = 100
        theta = np.linspace(-np.pi / 2, np.pi / 2, n_theta)
        phi = np.linspace(-np.pi, np.pi, n_phi)
        theta, phi = np.meshgrid(theta, phi)

        def fexp(base, exp):
            return np.sign(base) * (np.abs(base) ** exp)

        # Base (unit) superquadric shape
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        x = fexp(cos_theta, e1) * fexp(cos_phi, e2)
        y = fexp(cos_theta, e1) * fexp(sin_phi, e2)
        z = fexp(sin_theta, e1)

        points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=1)

        # Rotate to align Z-axis with principal direction
        axis_info = self.pcd.getAxis()
        if axis_info:
            target_axis = axis_info["direction"] / np.linalg.norm(axis_info["direction"])
            z_axis = np.array([0, 0, 1])
            v = np.cross(z_axis, target_axis)
            c = np.dot(z_axis, target_axis)

            if np.linalg.norm(v) < 1e-8:
                R = np.eye(3)
            else:
                vx = np.array([
                    [0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]
                ])
                R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (np.linalg.norm(v) ** 2))

            points = points @ R.T

        # Compute new bounding box of rotated unit superquadric
        min_p, max_p = points.min(axis=0), points.max(axis=0)
        unit_extent = max_p - min_p

        # Bounding box of real object
        bbox_min = self.pcd.getBoundingBox().get_min_bound()
        bbox_max = self.pcd.getBoundingBox().get_max_bound()
        target_extent = bbox_max - bbox_min

        # Compute scale factors for each axis
        scale = target_extent / unit_extent

        # Apply scaling
        points *= scale

        # Translate to centroid
        points += np.array([c1, c2, c3])

        # Reshape back to meshgrid
        x_final = points[:, 0].reshape(x.shape)
        y_final = points[:, 1].reshape(y.shape)
        z_final = points[:, 2].reshape(z.shape)

        return x_final, y_final, z_final



    def getSuperquadricAsPCD(self):
        x, y, z = self.modelValues

        points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    
    def getPCD(self):
        return self.pcd

    def updateSuperquadric(self):
        #update superquadric: for future development
        pass