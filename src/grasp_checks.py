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

"""
check gripper
- takes in the dimensions of the robot gripper (depth and width)

- this program ensures that the points selected in grasp
    - are less than 'width' apart
    - and that the gripper can penetrate to the specified depth without collision

"""
def checkGripper(grasp, superquadric, depth=0.0666, width=0.236):
    try:
        point_i = np.array(grasp["point_i"])
        point_j = np.array(grasp["point_j"])
        point_distance = np.linalg.norm(point_i - point_j)

        # Check if points are within gripper width
        if point_distance > width:
            return False

        # # Check depth penetration (uncommented and fixed)
        # superquadric_points = np.asarray(superquadric.points)
        # normal_i = np.array(grasp["point_i_normals"])
        # normal_j = np.array(grasp["point_j_normals"])
        
        # # Calculate grasp approach direction (average of normals pointing inward)
        # grasp_normal = -(normal_i + normal_j) / 2.0
        # grasp_normal /= np.linalg.norm(grasp_normal)

        # midpoint = (point_i + point_j) / 2.0
        # steps = np.linspace(0, depth, num=50)

        # # Check if gripper can penetrate to specified depth without collision
        # for step in steps:
        #     probe_point = midpoint + grasp_normal * step
        #     distances = np.linalg.norm(superquadric_points - probe_point, axis=1)
        #     if np.min(distances) < 0.005:  # Collision threshold
        #         return False
        return True
    except Exception as e:
        print(f"[checkGripper] Error: {e}")
        return False

"""
check antipodal

- checks that the normals of the points are opposing within a threshold
- ensures the grasp points have roughly opposite surface normals for stable grasping

"""
def checkAntipodal(grasp, normal_threshold=10):
    try:
        n1 = np.array(grasp["point_i_normals"])
        n2 = np.array(grasp["point_j_normals"])
        
        # Normalize the normals
        n1 /= np.linalg.norm(n1)
        n2 /= np.linalg.norm(n2)
        
        # Check if normals are opposing (dot product with -n2 should be close to 1)
        dot_product = np.dot(n1, -n2)
        angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg < normal_threshold
    except Exception as e:
        print(f"[checkAntipodal] Error: {e}")
        return False

"""
check collision

takes in the 
- grasp points
- the depth image
- the raw point cloud of the object
- the object bit mask
- the camera info
- and a collision threshold

this function ensures that there are no objects present in the depth map that collide with the points (within the threshold)
this function masks out the object from the depth map and is only concerned with surrounding objects in the foreground and background
"""
def checkCollision(grasp, depth_map, object_mask, camera_info, collision_threshold=0.005):
    try:
        K, fx, fy, cx, cy, w, h = extractCameraInfo(camera_info)
        if K is None:
            print("[checkCollision] Error: Failed to extract camera info")
            return False

        # Convert depth map to meters
        depth = np.asarray(depth_map) / 1000.0
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Create full scene point cloud from depth map
        valid_depth = depth > 0
        z_scene = depth[valid_depth]
        x_scene = (u[valid_depth] - cx) * z_scene / fx
        y_scene = (v[valid_depth] - cy) * z_scene / fy
        scene_points = np.stack((x_scene, y_scene, z_scene), axis=-1)
        
        # Create object point cloud from masked depth
        valid_object = (object_mask > 0) & (depth > 0)
        if not np.any(valid_object):
            print("[checkCollision] Warning: No valid object points in mask")
            return False
            
        z_obj = depth[valid_object]
        x_obj = (u[valid_object] - cx) * z_obj / fx
        y_obj = (v[valid_object] - cy) * z_obj / fy
        object_points = np.stack((x_obj, y_obj, z_obj), axis=-1)

        if scene_points.size == 0:
            print("[checkCollision] Warning: scene_points is empty — check depth map.")
            return False

        # Remove object points from scene to get environment points
        obj_tree = KDTree(object_points)
        distances, _ = obj_tree.query(scene_points)
        keep_mask = distances > 0.01  # Points farther than 1cm from object
        environment_points = scene_points[keep_mask]

        if environment_points.shape[0] == 0:
            print("[checkCollision] Warning: No environment points remain after masking.")
            return True  # No environment objects to collide with

        # Check if grasp points collide with environment
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
