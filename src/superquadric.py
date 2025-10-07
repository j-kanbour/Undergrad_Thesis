from pointCloudData import PointCloudData
import numpy as np
import open3d as o3d
from scipy.stats import kurtosis
import time
import re

class Superquadric:
    def __init__(self, pcd):
        #object_ID, class_name, raw_rgb, raw_depth=None, mask=None, camera_info=None
        self.print = lambda *args, **kwargs: print("Superquadric:", *args, **kwargs)

        #estimate values of e
        init_time = time.time()
        self.e1, self.e2 = self.estimateE(pcd)
        print(f"        e time: {time.time() - init_time:.3f}s")

        self.superquadric, self.pose = self.createSuperquadric(pcd, self.e1, self.e2)
        print(f"        SQ time: {time.time() - init_time:.3f}s")
        print(f"        Num Points: {len(self.superquadric.points)}")


    def estimateE(self, pcd):

        points = np.asarray(pcd.points)

        if points.shape[0] < 10:
            self.print("Not enough points to estimate shape reliably.")
            return 1.0, 1.0

        centroid = pcd.get_center()
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

    def createSuperquadric(self, pcd, e1, e2, res_u=128, res_v=256, point_percent=10):
        eta = np.linspace(-np.pi/2, np.pi/2, res_u)
        omega = np.linspace(-np.pi, np.pi, res_v, endpoint=False)
        Eta, Omega = np.meshgrid(eta, omega, indexing="ij")

        # Use the oriented bounding box (OBB) for size, rotation, and centre
        obb = pcd.get_minimal_oriented_bounding_box()
        a1, a2, a3 = obb.extent[0]/2, obb.extent[1]/2, obb.extent[2]/2
        R = obb.R  # 3x3 rotation (local -> world)
        
        # Visualize the pose of R
        import open3d as o3d
        
        # Create a coordinate frame at the origin with the rotation R
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(a1, a2, a3))
        frame.rotate(R, center=(0, 0, 0))
        frame.translate(obb.center)
        
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Rotation Pose Visualization")
        
        # Add geometries
        vis.add_geometry(pcd)
        vis.add_geometry(obb)
        vis.add_geometry(frame)
        
        # Run visualizer
        vis.run()
        vis.destroy_window()
        center = obb.center                      # world-space centre of the OBB

        def sgn(x):  # sign with zero preserved
            return np.sign(x + 1e-15)

        ce, se = np.cos(Eta), np.sin(Eta)
        co, so = np.cos(Omega), np.sin(Omega)

        # Local (superquadric/OBB) coordinates
        x = a1 * sgn(ce) * np.abs(ce)**e1 * sgn(co) * np.abs(co)**e2
        y = a2 * sgn(ce) * np.abs(ce)**e1 * sgn(so) * np.abs(so)**e2
        z = a3 * sgn(se) * np.abs(se)**e1

        V_local = np.stack([x, y, z], axis=-1).reshape(-1, 3)

        # Rotate into world frame using the OBB rotation, then translate to OBB centre
        V_world = (R @ V_local.T).T + center

        # Build mesh from world-space vertices
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
            vertices=o3d.utility.Vector3dVector(V_world),
            triangles=o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32)),
        )
        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()
        mesh.compute_vertex_normals()

        n_points = len(pcd.points) * (point_percent) // 100
        superquadricMesh = mesh.sample_points_poisson_disk(number_of_points=n_points, init_factor=5)
        superquadricMesh.estimate_normals()

        return superquadricMesh, R


    def getSuperquadricMesh(self):
        return self.superquadric
    
    def getSuperquadricPose(self):
        return self.pose