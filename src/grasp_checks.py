import numpy as np
from scipy.spatial import KDTree
import cv2
import traceback
import sys


def extractCameraInfo(camera_info):
    try:
        K = np.array(camera_info.K).reshape(3, 3)
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        w = camera_info.width
        h = camera_info.height
        return [K, fx, fy, cx, cy, w, h]
    except Exception as e:
        print(f"[extractCameraInfo] Error: {e}")
        return [None] * 7


def checkGripper(grasp, superquadric, depth=0.0666, width=0.236):
    try:
        point_i = np.array(grasp["point_i"])
        point_j = np.array(grasp["point_j"])
        point_distance = np.linalg.norm(point_i - point_j)

        if point_distance > width:
            return False

        superquadric = np.asarray(superquadric.points)
        normal_i = np.array(grasp["point_i_normals"])
        normal_j = np.array(grasp["point_j_normals"])
        grasp_normal = -(normal_i + normal_j) / 2.0
        grasp_normal /= np.linalg.norm(grasp_normal)

        midpoint = (point_i + point_j) / 2.0
        steps = np.linspace(0, depth, num=50)

        for step in steps:
            probe_point = midpoint + grasp_normal * step
            distances = np.linalg.norm(superquadric - probe_point, axis=1)
            if np.min(distances) < 0.005:
                return False
        return True
    except Exception as e:
        print(f"[checkGripper] Error: {e}")
        return False


def checkAntipodal(grasp, normal_threshold=10):
    try:
        n1 = np.array(grasp["point_i_normals"])
        n2 = np.array(grasp["point_j_normals"])
        n1 /= np.linalg.norm(n1)
        n2 /= np.linalg.norm(n2)
        dot_product = np.dot(n1, -n2)
        angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        return angle_deg < normal_threshold
    except Exception as e:
        print(f"[checkAntipodal] Error: {e}")
        return False

def checkCollision(grasp, depth_map, object_pcd, object_mask, camera_info, collision_threshold=0.005):
    try:
        K, fx, fy, cx, cy, w, h = extractCameraInfo(camera_info)

        depth = np.asarray(depth_map) / 1000.0
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        valid = (object_mask > 0) & (depth > 0)

        z = depth[valid]
        x = (u[valid] - cx) * z / fx
        y = (v[valid] - cy) * z / fy
        object_points = np.stack((x, y, z), axis=-1)

        scene_points = np.asarray(object_pcd.points)
        if object_points.size == 0:
            print("[checkCollision] Warning: object_points is empty — check depth map or mask.")
            return False

        obj_tree = KDTree(object_points)
        distances, _ = obj_tree.query(scene_points)
        keep_mask = distances > 0.01
        environment_points = scene_points[keep_mask]

        if environment_points.shape[0] == 0:
            print("[checkCollision] Warning: No environment points remain after masking.")
            return False

        env_tree = KDTree(environment_points)
        for grasp_point in [np.array(grasp["point_i"]), np.array(grasp["point_j"])]:
            dist, _ = env_tree.query(grasp_point)
            if dist < collision_threshold:
                return False
        return True

    except Exception as e:
        import traceback, sys
        tb = traceback.extract_tb(sys.exc_info()[2])[-1]
        print(f"[checkCollision] Error: {e} at line {tb.lineno} in {tb.filename}")
        return False




def checkOrientation(grasp, orientation, angle_threshold=15):
    try:
        angle = grasp["angle_to_xz"]

        if orientation in ['top', 'top2', 'front']:
            return abs(angle) < angle_threshold
        elif orientation == 'front-vertical':
            return abs(angle - 90) < angle_threshold
        else:
            return True
    except Exception as e:
        print(f"[checkOrientation] Error: {e}")
        return False


def checkAcrossFace(grasp, object_pcd, orientation, angle_threshold=45):
    try:
        if orientation in ['top', 'top2']:
            expected_dir = np.array([0, 0, -1])
        elif orientation in ['front', 'front-vertical']:
            expected_dir = np.array([0, -1, 0])
        else:
            return True

        n_i = np.array(grasp["point_i_normals"])
        n_j = np.array(grasp["point_j_normals"])
        n_i /= np.linalg.norm(n_i)
        n_j /= np.linalg.norm(n_j)
        expected_dir /= np.linalg.norm(expected_dir)

        angle_i = np.degrees(np.arccos(np.clip(np.dot(n_i, expected_dir), -1.0, 1.0)))
        angle_j = np.degrees(np.arccos(np.clip(np.dot(n_j, expected_dir), -1.0, 1.0)))

        return angle_i <= angle_threshold and angle_j <= angle_threshold
    except Exception as e:
        print(f"[checkAcrossFace] Error: {e}")
        return False


def checkPose():
    try:
        pass  # Placeholder
    except Exception as e:
        print(f"[checkPose] Error: {e}")
        return False


def checkForceClosure():
    try:
        pass  # Placeholder
    except Exception as e:
        print(f"[checkForceClosure] Error: {e}")
        return False
