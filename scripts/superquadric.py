from pointCloudData import PointCloudData
import numpy as np
import open3d as o3d
from scipy.stats import kurtosis

class Superquadric:
    def __init__(self, object_ID, class_name, input_type, raw_data_1, raw_depth=None, raw_mask=None, camera_info=None):
        self.print = lambda *args, **kwargs: print("Superquadric:", *args, **kwargs)
        self.object_ID = object_ID
        self.class_name = class_name

        self.pcd = PointCloudData(object_ID, input_type, raw_data_1, raw_depth, raw_mask, camera_info)

        self.e1, self.e2 = self.estimateE()
        self.modelValues = self.createSuperquadric()

    def estimateE(self):
        points = np.asarray(self.pcd.getPCD().points)

        if points.shape[0] < 10:
            self.print("Not enough points to estimate shape reliably.")
            return 1.0, 1.0

        centroid = self.pcd.getCentroid()
        centered_points = points - centroid

        krt = kurtosis(centered_points, axis=0, fisher=True, bias=False)

        e1 = np.clip(1 + (krt[2] - 3) * 0.1, 0.3, 2.0)
        e2 = np.clip(1 + ((krt[0] + krt[1]) / 2 - 3) * 0.1, 0.3, 2.0)

        self.print(f"Estimated e1: {e1:.3f}, e2: {e2:.3f}")
        return e1, e2

    def createSuperquadric(self):
        c1, c2, c3 = self.pcd.getCentroid()
        e1, e2 = self.e1, self.e2

        n_theta = 100
        n_phi = 100
        theta = np.linspace(-np.pi / 2, np.pi / 2, n_theta)
        phi = np.linspace(-np.pi, np.pi, n_phi)
        theta, phi = np.meshgrid(theta, phi)

        def fexp(base, exp):
            return np.sign(base) * (np.abs(base) ** exp)

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        x = fexp(cos_theta, e1) * fexp(cos_phi, e2)
        y = fexp(cos_theta, e1) * fexp(sin_phi, e2)
        z = fexp(sin_theta, e1)

        points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=1)

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

        min_p, max_p = points.min(axis=0), points.max(axis=0)
        unit_extent = max_p - min_p

        bbox_min = self.pcd.getBoundingBox().get_min_bound()
        bbox_max = self.pcd.getBoundingBox().get_max_bound()
        target_extent = bbox_max - bbox_min

        scale = target_extent / unit_extent
        points *= scale

        # 🌟 Align base of superquadric with lowest point in point cloud
        pcd_points = np.asarray(self.pcd.getPCD().points)
        pcd_min_z = np.min(pcd_points[:, 2])
        superquadric_min_z = np.min(points[:, 2])

        z_offset = pcd_min_z - superquadric_min_z
        xy_mean = np.mean(points[:, :2], axis=0)
        translation_offset = np.array([
            c1 - xy_mean[0],
            c2 - xy_mean[1]+0.01,
            z_offset - 0.01
        ])
        points += translation_offset

        x_final = points[:, 0].reshape(x.shape)
        y_final = points[:, 1].reshape(y.shape)
        z_final = points[:, 2].reshape(z.shape)

        return x_final, y_final, z_final

    def alignWithICP(self, threshold=0.02):
        source = self.getSuperquadricAsPCD()
        target = self.pcd.getPCD()

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
        return aligned_pcd



    def getSuperquadricAsPCD(self):
        x, y, z = self.modelValues
        points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    def getPCD(self):
        return self.pcd

    def updateSuperquadric(self):
        pass
