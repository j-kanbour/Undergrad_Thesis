# sq_fixed_eps.py  –  superquadric fit with class-fixed ε
# -----------------------------------------------------------
# 1) mirror  2) PCA  3) optimise scales (ε fixed)  4) sample
# -----------------------------------------------------------

from __future__ import annotations
import re, sys, pathlib
import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from typing import Dict, Tuple

# ───────────── helpers: mirroring & PCA ─────────────────────
def mirror_cloud(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    xyz = np.asarray(pcd.points)
    mirrored = o3d.geometry.PointCloud()
    mirrored.points = o3d.utility.Vector3dVector(xyz * np.array([-1, 1, 1]))
    if pcd.has_colors():
        mirrored.colors = pcd.colors
    return mirrored

def pca_align(pcd: o3d.geometry.PointCloud):
    xyz = np.asarray(pcd.points)
    centre = xyz.mean(0)
    xyz_c  = xyz - centre
    eigv, eigvec = np.linalg.eigh(np.cov(xyz_c.T))
    R = eigvec[:, eigv.argsort()[::-1]]
    return (xyz_c @ R, R, centre)        # aligned pts, rot, centre

# ───────────── class → fixed ε mapping ──────────────────────
def fixed_eps(class_name: str) -> Tuple[float,float]:
    cn = (class_name or "").lower()
    if re.search(r"can|cup|mug", cn):   return 1.0, 1.0   # cylinder
    if re.search(r"box", cn):           return 0.3, 0.3   # cuboid
    if re.search(r"ball|sphere", cn):   return 1.0, 1.0   # sphere
    if re.search(r"bowl|plate", cn):    return 0.6, 0.6
    return 1.0, 1.0                                   # fallback

# ───────────── residual (radial-weighted) ───────────────────
def _sq_F(a1,a2,a3,e1,e2, xyz):
    x,y,z = xyz[:,0]/a1, xyz[:,1]/a2, xyz[:,2]/a3
    f = (np.abs(x)**(2/e2)+np.abs(y)**(2/e2))**(e2/e1) + np.abs(z)**(2/e1) - 1
    return f

def _res_scales(a, xyz, e1, e2):
    return np.linalg.norm(xyz,axis=1) * _sq_F(a[0],a[1],a[2], e1,e2, xyz)

# ───────────── scale optimiser (ε fixed) ────────────────────
def fit_scales(xyz: np.ndarray, e1: float, e2: float):
    a0 = xyz.ptp(0)/2
    res = least_squares(_res_scales, a0, args=(xyz, e1, e2),
                        method='lm', max_nfev=200)
    mse = res.cost / xyz.shape[0]
    return res.x, mse

# ───────────── dense SQ sampler ─────────────────────────────
def sample_sq(a1,a2,a3,e1,e2, n_th=72, n_ph=144):
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
def fit_superquadric_cloud(pcd: o3d.geometry.PointCloud, *,
                           class_name: str = "",
                           mirror_first: bool = True):
    # 1. symmetry completion
    pcd_full = pcd + mirror_cloud(pcd) if mirror_first else pcd

    # 2. PCA align
    xyz_aligned, R, centre = pca_align(pcd_full)

    # 3. fixed ε from class label
    e1,e2 = fixed_eps(class_name)

    # 4. optimise scales only
    scales, mse = fit_scales(xyz_aligned, e1, e2)
    a1,a2,a3 = scales

    # 5. sample & transform back
    sq = sample_sq(a1,a2,a3,e1,e2)
    sq.points = o3d.utility.Vector3dVector(np.asarray(sq.points) @ R.T + centre)

    params = dict(a1=a1,a2=a2,a3=a3,e1=e1,e2=e2,rms=np.sqrt(mse))
    return sq, params

# CLI test
if __name__ == "__main__":
    if len(sys.argv)<3:
        print("Usage: python sq_fixed_eps.py cloud.pcd class_label"); sys.exit(1)
    cloud = o3d.io.read_point_cloud(sys.argv[1])
    sq, prm = fit_superquadric_cloud(cloud, class_name=sys.argv[2])
    print(prm)
    cloud.paint_uniform_color([0.8,0.6,0.1]); sq.paint_uniform_color([0.1,0.4,0.9])
    o3d.visualization.draw_geometries([cloud,sq])
