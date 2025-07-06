# prealign.py
# ─────────────────────────────────────────────────────────────
# Implements Steps 3 & 4 of the Makhal-Thomas pipeline.
#   • PCA-based or AABB-midpoint centre & orientation guess
#   • Returns 4×4 affine, axis variances, and the pre-aligned cloud
#
# Dependencies: open3d >= 0.18, numpy
# ----------------------------------------------------------------–

from __future__ import annotations
import numpy as np
import open3d as o3d


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────
def _aabb_centre(pts: np.ndarray) -> np.ndarray:
    """Axis-aligned bounding-box midpoint (robust to density bias)."""
    return 0.5 * (pts.min(0) + pts.max(0))


def _pca(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Principal component directions & variances.

    Returns
    -------
    vecs : (3,3) eigenvectors ordered λ0≥λ1≥λ2
    vals : (3,)  eigenvalues
    """
    centred = pts - pts.mean(0)
    cov = np.cov(centred.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vecs[:, order], vals[order]


def _build_affine(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Return 4×4 homogeneous matrix."""
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _apply_T(pcd: o3d.geometry.PointCloud, T: np.ndarray) -> o3d.geometry.PointCloud:
    """Return a *copy* of pcd transformed by T."""
    pcd_out = o3d.geometry.PointCloud(pcd)  # shallow copy of geometry
    pcd_out.transform(T)                    # Open3D uses 4×4
    return pcd_out


# ─────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────
def pre_align(pcd: o3d.geometry.PointCloud,
              method: str = "pca",
              pre_align_axis: int = 0
              ) -> tuple[np.ndarray, np.ndarray, o3d.geometry.PointCloud]:
    """
    Parameters
    ----------
    pcd : Open3D PointCloud
        Merged (original+mirrored) object cloud.
    method : {"pca", "bbox"}, default "pca"
        Pose-initialisation strategy.
    pre_align_axis : 0|1|2, default 0
        Which principal axis you want aligned to +X for this trial.

    Returns
    -------
    T_pre : (4,4) ndarray float32
        Affine that centres and orients the cloud.
    variances : (3,) ndarray
        Axis variances in the pre-aligned frame (feed to initial a₁..a₃).
    cloud_out : Open3D PointCloud
        Copy of input cloud after applying T_pre   (Step 4).
    """
    if pcd.is_empty():
        raise ValueError("Input cloud is empty")

    pts = np.asarray(pcd.points, dtype=np.float32)

    # ── 1. Centre estimate ─────────────────────────────────────────────
    if method == "bbox":
        centre = _aabb_centre(pts)
        R = np.eye(3)                       # no rotation
        variances = np.var(pts - centre, axis=0)          # just for scale guess
    elif method == "pca":
        centre = pts.mean(0)
        R_pca, eigvals = _pca(pts)
        # swap columns so requested axis becomes +X
        R_swap = R_pca.copy()
        R_swap[:, [0, pre_align_axis]] = R_swap[:, [pre_align_axis, 0]]
        R = R_swap
        variances = eigvals[[pre_align_axis, (pre_align_axis+1)%3, (pre_align_axis+2)%3]]
    else:
        raise ValueError("method must be 'pca' or 'bbox'")

    # ── 2. Build transform: first translate to origin, then rotate ─────
    T_trans = _build_affine(np.eye(3), -centre)
    T_rot   = _build_affine(R, np.zeros(3))
    T_pre   = T_rot @ T_trans                         # note order

    # ── 3. Apply to cloud (Step 4) ─────────────────────────────────────
    cloud_out = _apply_T(pcd, T_pre)

    return T_pre, variances.astype(np.float32), cloud_out


# ─────────────────────────────────────────────────────────────
# Quick visual sanity-check
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, pathlib
    if len(sys.argv) < 2:
        print("Usage: python prealign.py <object_segment.pcd>")
        sys.exit(1)

    p = pathlib.Path(sys.argv[1])
    cloud_in = o3d.io.read_point_cloud(str(p))

    T, var, cloud_aligned = pre_align(cloud_in, method="pca", pre_align_axis=0)
    print("T_pre:\n", T)
    print("Axis variances:", var)

    cloud_in.paint_uniform_color([0.8, 0.6, 0.1])
    cloud_aligned.paint_uniform_color([0.1, 0.4, 0.9])
    o3d.visualization.draw_geometries([cloud_in], window_name="Input")
    o3d.visualization.draw_geometries([cloud_aligned], window_name="After pre-align")
