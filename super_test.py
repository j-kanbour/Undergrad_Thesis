import numpy as np
import open3d as o3d

# your mesh builder from before
def superquadric_mesh(a1, a2, a3, e1, e2, res_u=128, res_v=256):
    eta = np.linspace(-np.pi/2, np.pi/2, res_u)
    omega = np.linspace(-np.pi, np.pi, res_v, endpoint=False)
    Eta, Omega = np.meshgrid(eta, omega, indexing="ij")

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
    return mesh

# Rodrigues helpers to apply pose if you have it from a fit
def rodrigues_to_R(r):
    theta = np.linalg.norm(r)
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

def apply_pose_to_pcd(pcd, t, r):
    R = rodrigues_to_R(r)
    P = np.asarray(pcd.points)
    P = P @ R.T + t
    pcd.points = o3d.utility.Vector3dVector(P)
    if pcd.has_normals():
        N = np.asarray(pcd.normals)
        N = N @ R.T
        pcd.normals = o3d.utility.Vector3dVector(N)
    return pcd

# Method A: sample points from the mesh
def superquadric_pointcloud_from_mesh(a1, a2, a3, e1, e2, n_points=50000, res_u=160, res_v=320):
    mesh = superquadric_mesh(a1, a2, a3, e1, e2, res_u, res_v)
    # Poisson disk gives nice blue noise distribution
    pcd = mesh.sample_points_poisson_disk(number_of_points=n_points, init_factor=5)
    # If you prefer a faster even triangle based sampler:
    # pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    pcd.estimate_normals()
    return pcd

# Example usage
if __name__ == "__main__":
    # can like superquadric
    a1, a2, a3 = 0.035, 0.035, 0.06
    e1, e2 = 0.1,1

    pcd_a = superquadric_pointcloud_from_mesh(a1, a2, a3, e1, e2, n_points=40000)
    pcd_a.paint_uniform_color([0.2, 0.6, 1.0])

    # If you have fitted pose pars = {'t':..., 'r':..., 'a1':..., etc}
    # pcd_a = apply_pose_to_pcd(pcd_a, pars['t'], pars['r'])

    o3d.visualization.draw_geometries([pcd_a])


