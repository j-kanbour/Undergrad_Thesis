import numpy as np
from scipy.spatial import KDTree


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
- returns a score out of 100 based on how well the grasp fits within gripper constraints
- considers both width constraint and depth penetration feasibility
"""
def checkGripper(grasp, superquadric, depth=0.0666, width=0.236):
    try:
        point_i = np.array(grasp["point_i"])
        point_j = np.array(grasp["point_j"])
        point_distance = np.linalg.norm(point_i - point_j)

        # Score based on gripper width constraint
        if point_distance > width:
            # Penalize based on how much it exceeds the width
            excess_ratio = point_distance / width
            width_score = max(0, 100 - (excess_ratio - 1) * 200)  # Steep penalty for exceeding width
        else:
            # Reward points that are close to optimal width (around 60-80% of max width)
            optimal_ratio = point_distance / width
            if 0.6 <= optimal_ratio <= 0.8:
                width_score = 100
            elif optimal_ratio < 0.6:
                # Too narrow - linear penalty
                width_score = 100 * (optimal_ratio / 0.6)
            else:
                # Too wide but within limits - linear penalty
                width_score = 100 * (1 - (optimal_ratio - 0.8) / 0.2)
        
        # Depth penetration score (simplified - assume good penetration for now)
        depth_score = 100
        
        # Could add actual depth penetration check here:
        # superquadric_points = np.asarray(superquadric.points)
        # normal_i = np.array(grasp["point_i_normals"])
        # normal_j = np.array(grasp["point_j_normals"])
        # ... collision checking logic ...
        
        # Combined score (weighted average)
        final_score = (width_score * 0.8 + depth_score * 0.2)
        return max(0, min(100, final_score))
        
    except Exception as e:
        print(f"[checkGripper] Error: {e}")
        return 0

"""
check antipodal
- checks that the normals of the points are opposing within a threshold
- returns a score out of 100 based on how well the normals oppose each other
"""
def checkAntipodal(grasp, normal_threshold=10):
    try:
        n1 = np.array(grasp["point_i_normals"])
        n2 = np.array(grasp["point_j_normals"])
        
        # Normalize the normals
        n1 /= np.linalg.norm(n1)
        n2 /= np.linalg.norm(n2)
        
        # Calculate dot product directly (should be close to -1 for opposing normals)
        dot_product = np.dot(n1, n2)
        return 100 * dot_product * (-1)
        # # Debug prints
        # print(f"[checkAntipodal] Normal 1: {n1}")
        # print(f"[checkAntipodal] Normal 2: {n2}")
        # print(f"[checkAntipodal] Dot product: {dot_product:.4f}")
        
        # # For opposing normals, dot product should be close to -1
        # # Convert to angle for scoring
        # angle_rad = np.arccos(np.clip(abs(dot_product), 0.0, 1.0))
        # angle_deg = np.degrees(angle_rad)
        
        # # If dot product is positive, normals are pointing in same direction (bad)
        # # If dot product is negative, normals are opposing (good)
        # if dot_product > 0:
        #     # Same direction - penalize heavily
        #     score = max(0, 20 * (1 - dot_product))
        # else:
        #     # Opposing direction - score based on how close to -1
        #     opposition_quality = abs(dot_product)  # How close to -1 (ranges from 0 to 1)
            
        #     # Convert to angle for threshold comparison
        #     if angle_deg <= normal_threshold:
        #         # Within threshold - linear scoring
        #         score = 100 * opposition_quality
        #     else:
        #         # Beyond threshold - exponential decay
        #         excess_angle = angle_deg - normal_threshold
        #         score = max(0, 100 * opposition_quality * np.exp(-excess_angle / 30))
        
        # print(f"[checkAntipodal] Angle: {angle_deg:.2f} degrees, Score: {score:.2f}")
        # return max(0, min(100, score))
        
    except Exception as e:
        print(f"[checkAntipodal] Error: {e}")
        return 0

"""
check collision
- returns a score out of 100 based on collision risk
- higher score means lower collision risk
- works with already-masked depth data (depth_masked)
"""
def checkCollision(grasp, depth_masked, camera_info, collision_threshold=0.03):
    try:
        K, fx, fy, cx, cy, w, h = extractCameraInfo(camera_info)
        if K is None:
            print("[checkCollision] Error: Failed to extract camera info")
            return 0

        # Convert masked depth to meters (assuming input is in mm)
        depth = np.asarray(depth_masked)
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Create point cloud from masked depth (only object points)
        valid_depth = depth > 0
        if not np.any(valid_depth):
            print("[checkCollision] Warning: No valid depth points in masked depth")
            return 0
            
        z_obj = depth[valid_depth]
        x_obj = (u[valid_depth] - cx) * z_obj / fx
        y_obj = (v[valid_depth] - cy) * z_obj / fy
        object_points = np.stack((x_obj, y_obj, z_obj), axis=-1)

        if object_points.size == 0:
            print("[checkCollision] Warning: object_points is empty")
            return 0

        # Build KDTree for object points
        obj_tree = KDTree(object_points)
        
        # Check distance from grasp points to nearest object surface
        min_distance = float('inf')
        grasp_points = [np.array(grasp["point_i"]), np.array(grasp["point_j"])]
        
        for grasp_point in grasp_points:
            dist, _ = obj_tree.query(grasp_point)
            min_distance = min(min_distance, dist)
        
        # Score based on minimum distance to object surface
        # Higher distance = higher score (less collision risk)
        if min_distance >= collision_threshold * 3:  # Safe distance
            score = 100
        elif min_distance >= collision_threshold:
            # Linear interpolation between threshold and 3x threshold
            score = 50 + 50 * (min_distance - collision_threshold) / (collision_threshold * 2)
        else:
            # Close to collision - exponential decay
            score = 50 * (min_distance / collision_threshold)
        
        return max(0, min(100, score))

    except Exception as e:
        import traceback, sys
        tb = traceback.extract_tb(sys.exc_info()[2])[-1]
        print(f"[checkCollision] Error: {e} at line {tb.lineno} in {tb.filename}")
        return 0

def checkOrientation(grasp, orientation, angle_threshold=15):
    try:
        angle = grasp["angle_to_xz"]

        if orientation in ['top', 'top2', 'front']:
            target_angle = 0
        elif orientation == 'front-vertical':
            target_angle = 90
        else:
            return 100  # No orientation constraint
        
        angle_diff = abs(angle - target_angle)
        
        # Score based on angle difference
        if angle_diff <= angle_threshold:
            score = 100 * (1 - angle_diff / angle_threshold)
        else:
            # Beyond threshold - exponential decay
            excess_angle = angle_diff - angle_threshold
            score = max(0, 100 * np.exp(-excess_angle / 30))
        
        return max(0, min(100, score))
        
    except Exception as e:
        print(f"[checkOrientation] Error: {e}")
        return 0

def checkAcrossFace(grasp, object_pcd, orientation, angle_threshold=45):
    try:
        if orientation in ['top', 'top2']:
            expected_dir = np.array([0, 0, -1])
        elif orientation in ['front', 'front-vertical']:
            expected_dir = np.array([0, -1, 0])
        else:
            return 100  # No constraint

        n_i = np.array(grasp["point_i_normals"])
        n_j = np.array(grasp["point_j_normals"])
        n_i /= np.linalg.norm(n_i)
        n_j /= np.linalg.norm(n_j)
        expected_dir /= np.linalg.norm(expected_dir)

        angle_i = np.degrees(np.arccos(np.clip(np.dot(n_i, expected_dir), -1.0, 1.0)))
        angle_j = np.degrees(np.arccos(np.clip(np.dot(n_j, expected_dir), -1.0, 1.0)))

        # Score for each point
        def angle_score(angle):
            if angle <= angle_threshold:
                return 100 * (1 - angle / angle_threshold)
            else:
                excess_angle = angle - angle_threshold
                return max(0, 100 * np.exp(-excess_angle / 30))
        
        score_i = angle_score(angle_i)
        score_j = angle_score(angle_j)
        
        # Combined score (both points must be good)
        final_score = min(score_i, score_j)
        return max(0, min(100, final_score))
        
    except Exception as e:
        print(f"[checkAcrossFace] Error: {e}")
        return 0

def checkPose(grasp):
    try:
        # Placeholder implementation - could check pose stability, reachability, etc.
        # For now, return a neutral score
        return 75
    except Exception as e:
        print(f"[checkPose] Error: {e}")
        return 0

def checkForceClosure(grasp):
    try:
        # Placeholder implementation - could check force closure conditions
        # For now, return a neutral score
        return 75
    except Exception as e:
        print(f"[checkForceClosure] Error: {e}")
        return 0

def calculateOverallGraspScore(grasp, superquadric, depth_map, object_mask, camera_info, orientation, weights=None):
    """
    Calculate overall grasp score as weighted combination of all individual scores
    
    Args:
        grasp: grasp dictionary with points, normals, etc.
        superquadric: superquadric object
        depth_map: depth image
        object_mask: binary mask of the object
        camera_info: camera parameters
        orientation: grasp orientation ('top', 'front', etc.)
        weights: dictionary of weights for each score component
    
    Returns:
        overall_score: float between 0-100
        individual_scores: dictionary of individual scores
    """
    try:
        # Default weights - can be adjusted based on application requirements
        if weights is None:
            weights = {
                'gripper': 0.25,
                'antipodal': 0.25,
                'collision': 0.20,
                'orientation': 0.15,
                'across_face': 0.10,
                'pose': 0.03,
                'force_closure': 0.02
            }
        
        # Calculate individual scores
        individual_scores = {
            'gripper': checkGripper(grasp, superquadric),
            'antipodal': checkAntipodal(grasp),
            'collision': checkCollision(grasp, depth_map, object_mask, camera_info),
            'orientation': checkOrientation(grasp, orientation),
            'across_face': checkAcrossFace(grasp, None, orientation),
            'pose': checkPose(grasp),
            'force_closure': checkForceClosure(grasp)
        }
        
        # Calculate weighted overall score
        overall_score = sum(weights[key] * individual_scores[key] for key in weights.keys())
        
        return overall_score, individual_scores
        
    except Exception as e:
        print(f"[calculateOverallGraspScore] Error: {e}")
        return 0, {}