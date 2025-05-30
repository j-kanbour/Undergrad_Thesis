import numpy as np
import open3d as o3d
import cv2
import json
import os

def create_target_point_cloud(rgb_path, depth_path, mask_path, scene_info_json):
    # --- Load files ---
    assert os.path.exists(rgb_path), f"RGB image not found: {rgb_path}"
    assert os.path.exists(depth_path), f"Depth image not found: {depth_path}"
    assert os.path.exists(mask_path), f"Mask image not found: {mask_path}"
    assert os.path.exists(scene_info_json), f"Camera info not found: {scene_info_json}"

    rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Debug shapes
    print(f"RGB shape: {rgb.shape}, dtype: {rgb.dtype}")
    print(f"Depth shape: {depth.shape}, dtype: {depth.dtype}")
    print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")

    # --- Load camera intrinsics ---
    with open(scene_info_json, 'r') as f:
        scene_info = json.load(f)["0"]
        K = np.array(scene_info["cam_K"]).reshape(3, 3)
        depth_scale = float(scene_info.get("depth_scale", 1.0))

    # Resize mask if needed
    if mask.shape != depth.shape:
        print("Resizing mask to match depth resolution...")
        mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)

    # --- Validate depth type and scale ---
    if depth.dtype == np.float32 or depth.dtype == np.float64:
        print("Detected depth in metres, setting depth_scale = 1.0")
        depth_scale = 1.0
    elif depth.dtype == np.uint16 or depth.dtype == np.uint32:
        print("Assuming depth in millimetres, setting depth_scale = 1000.0")
        depth_scale = 1000.0
    else:
        raise ValueError(f"Unsupported depth dtype: {depth.dtype}")

    print("Depth min:", np.min(depth), "max:", np.max(depth))

    # --- Apply combined mask: target + valid depth ---
    valid_mask = (mask > 0) & (depth > 0)
    print("Valid points in mask:", np.count_nonzero(valid_mask))

    if np.count_nonzero(valid_mask) == 0:
        print("No valid masked points found — check depth or mask alignment.")
        return

    depth_masked = np.where(valid_mask, depth, 0)
    rgb_masked = np.zeros_like(rgb)
    rgb_masked[valid_mask] = rgb[valid_mask]

    # --- Create Open3D RGBD image ---
    depth_o3d = o3d.geometry.Image(depth_masked.astype(np.uint16))
    color_o3d = o3d.geometry.Image(rgb_masked.astype(np.uint8))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=depth_scale,
        depth_trunc=3.0,
        convert_rgb_to_intensity=False
    )

    # --- Define intrinsics ---
    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

    # --- Generate point cloud ---
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])

    # --- Visualise or report ---
    if len(pcd.points) == 0:
        print("Point cloud is empty. Double check image alignment and masking.")
        return

    print("Generated point cloud with", len(pcd.points), "points.")
    o3d.visualization.draw_geometries([pcd], window_name="Target Object Point Cloud")


# Example usage
create_target_point_cloud(
    rgb_path="rgb and depth data/000001/rgb/000000.png",
    depth_path="rgb and depth data/000001/depth/000000.png",
    mask_path="rgb and depth data/000001/mask_visib/000000_000000.png",
    scene_info_json="rgb and depth data/000001/scene_camera.json"
)
