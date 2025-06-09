from pointCloudData import PointCloudData
import numpy as np
import open3d as o3d
from scipy.stats import kurtosis

class Superquadric:
    def __init__(self, object_ID, class_name, input_type, raw_data_1, raw_depth=None, raw_mask=None, camera_info=None):
        self.print = lambda *args, **kwargs: print("Superquadric:", *args, **kwargs)

        self.object_ID = object_ID
        self.class_name = class_name

        self.primitive_shape = self.findPrimitive()

        #built point cloud from raw data
        self.pcd = PointCloudData(object_ID, input_type, raw_data_1, raw_depth, raw_mask, camera_info)

        #estimate values of e
        self.e1, self.e2 = self.estimateE()

        #determine superquadric values
        self.modelValues = self.createSuperquadric()

    def findPrimitive(self):
        match self.class_name:
            case "Can":
                return "CYLINDER"
            case "Bottle":
                return "CYLINDER"
            case "Box":
                return "CUBOID"
            case "Ball":
                return "SPHERE"
            case _:
                return "GENERAL"
            

        """NOTE: Need to account for many more cases and multiple primitive objects"""
            

    def estimateE(self):

        """TODO: Check the accuracy of this method, is there any better ones in terms of accuracy and speed"""

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

    def createSuperquadric(self):
        """Generate a 3D shape based on the specified type (e.g., sphere, cylinder, cone)."""
        
        # Get the centroid and estimated e values
        c1, c2, c3 = self.pcd.getCentroid()
        e1, e2 = self.e1, self.e2

        # Control resolution
        n_theta = 100  # Resolution in the vertical direction
        n_phi = 100    # Resolution in the circular direction
        
        # Define the parameterization space
        theta = np.linspace(-np.pi / 2, np.pi / 2, n_theta)  # Vertical angle for spherical and cone/cylinder height
        phi = np.linspace(-np.pi, np.pi, n_phi)  # Circular angle for spherical and cylindrical symmetry
        
        # Create mesh grid
        theta, phi = np.meshgrid(theta, phi)
        
        # Generate shape based on selected type
        if self.primitive_shape == "SPHERE":
            r = 1  # Radius for the sphere
            x = r * np.cos(theta) * np.cos(phi)
            y = r * np.cos(theta) * np.sin(phi)
            z = r * np.sin(theta)
            
        elif self.primitive_shape == "CYLINDER":
            r = 1  # Fixed radius for the cylinder
            z = np.linspace(-1, 1, n_theta)  # Height range from -1 to 1
            phi = np.linspace(0, 2 * np.pi, n_phi)  # Full circle for cylinder
            z, phi = np.meshgrid(z, phi)
            
            x = r * np.cos(phi)  # Cylinder x-coordinate
            y = r * np.sin(phi)  # Cylinder y-coordinate
            
        elif self.primitive_shape == "CONE":
            r_max = 1  # Max radius at the base
            z = np.linspace(-1, 1, n_theta)  # Height from -1 to 1
            phi = np.linspace(0, 2 * np.pi, n_phi)  # Full circular range
            z, phi = np.meshgrid(z, phi)
            
            # Scaling radius as z increases
            r = r_max * (1 - (z / max(z)))  # Linear decrease of radius with height
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            
        else:
            raise ValueError(f"Unsupported shape type: {self.primitive_shape}")
        
        points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=1)

        # Align shape with point cloud's axis and bounding box
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

        # Scale shape based on bounding box of the point cloud
        min_p, max_p = points.min(axis=0), points.max(axis=0)
        unit_extent = max_p - min_p

        bbox_min = self.pcd.getBoundingBox().get_min_bound()
        bbox_max = self.pcd.getBoundingBox().get_max_bound()
        target_extent = bbox_max - bbox_min

        scale = target_extent / unit_extent
        points *= scale

        # Align shape with the lowest point of the point cloud
        pcd_points = np.asarray(self.pcd.getPCD().points)
        pcd_min_z = np.min(pcd_points[:, 2])
        shape_min_z = np.min(points[:, 2])

        z_offset = pcd_min_z - shape_min_z
        xy_mean = np.mean(points[:, :2], axis=0)
        translation_offset = np.array([
            c1 - xy_mean[0],
            c2 - xy_mean[1] + 0.01,
            z_offset - 0.01
        ])
        points += translation_offset

        # Reshape the points back to the mesh grid
        x_final = points[:, 0].reshape(x.shape)
        y_final = points[:, 1].reshape(y.shape)
        z_final = points[:, 2].reshape(z.shape)

        return x_final, y_final, z_final

    def alignWithICP(self, threshold=0.02):
        
        """NOTE: WTF is going on here"""

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
        
        """Builds PCD based on superquadric values"""

        x, y, z = self.modelValues
        points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    def getPCD(self):
        return self.pcd

    def updateSuperquadric(self):
        pass
