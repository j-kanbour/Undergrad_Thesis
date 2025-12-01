#!/usr/bin/env python3.8

"""
    pointCloudData: extracts data from raw rgb and depth input, 
                    applies a mask and removes outliers
                    also can segment the object model into multiple planes using RANSAC

    Input:
        raw_rgb: raw rgb image from camera
        raw_depth: raw depth image from camera
        mask: mask of the object to extract from rgb and depth image
        camera_info: camera info from camera
        nearest_neighbor: number of nearest neighbors to consider for outlier removal
        distance_thresh: distance threshold for RANSAC plane segmentation
        semght_threshold: threshold to stop segmentation based on remaining points
        debug: if True, print debug information

    Output:
        pcd: point cloud of the masked object
        cloud_segments: list of point cloud segments from RANSAC

    Developed by: Jayden Kanbour as part of undergraduate thesis for UNSW Computer Science and Engineering
    Date: 26th November 2025
    Email: jkanbour1@gmail.com
    UNSW Student Id: z5316799

"""

import cv2
import numpy as np
import open3d as o3d
import time
import copy
import os
import json

class PointCloudData:

    def __init__(self, raw_rgb, raw_depth, mask, camera_info, nearest_neighbor=500, distance_thresh=0.005, semght_threshold=10, debug=False):

        #convert ROS images to CV2

        assert os.path.exists(raw_rgb), f"RGB image not found: {raw_rgb}"
        assert os.path.exists(raw_depth), f"Depth image not found: {raw_depth}"
        assert os.path.exists(mask), f"Mask image not found: {mask}"

        for path, name in [(raw_rgb, "RGB"), (raw_depth, "Depth"), (mask, "Mask")]:
            assert os.path.exists(path), f"{name} image not found: {path}"

        # Load images
        self.raw_rgb = o3d.io.read_image(raw_rgb)
        self.raw_depth = o3d.io.read_image(raw_depth)
        self.mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)

        self.debug = debug
        self.camera_info = self.extractCameraInfo(camera_info)
        self.nearest_neighbout = nearest_neighbor
        self.distance_thresh = distance_thresh
        self.segmentation_threshold = semght_threshold

        start_time = time.time()

        #convert extracted data to point cloud
        self.pcd = self.covertToPCD()

        if self.debug:
            print(f"pointCloudData: Point cloud generated within {time.time() - start_time:.3f}s")

    def extractCameraInfo(self, camera_info):

        with open(camera_info, "r") as f:
            camera_info = json.load(f)

        cam_data = camera_info["0"]
        fx, fy = cam_data["cam_K"][0], cam_data["cam_K"][4]
        cx, cy = cam_data["cam_K"][2], cam_data["cam_K"][5]
        
        # Construct the full K matrix
        K = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]])
        
        # Get width and height from the camera info
        # This depends on your data structure - adjust as needed
        w = cam_data.get("width", int(cx * 2))  # Default to cx*2 if not provided
        h = cam_data.get("height", int(cy * 2))  # Default to cy*2 if not provided
        
        return [K, fx, fy, cx, cy, w, h]
    #remove outliers from, may not need if mask is good
    def removeOutliers(self, pcd):
        """
            Removes outliers based on nearest neighbour algorithm
        """
        try:
            if pcd.is_empty():
                print("pointCloudData [removeOutliers] Warning: Provided point cloud is empty.")
                return pcd
            
            # Efficient parameters
            _, ind = pcd.remove_statistical_outlier(nb_neighbors=self.nearest_neighbout, std_ratio=0.25)

            return pcd.select_by_index(ind)

        except Exception as e:
            print(f"pointCloudData [removeOutliers] Error: {e}")
            return None

    def covertToPCD(self):
    
        try:
            #Handle Mask

            # Camera intrinsics
            fx = self.camera_info[1]
            fy = self.camera_info[2]
            cx = self.camera_info[3]
            cy = self.camera_info[4]
            w = self.camera_info[5]
            h = self.camera_info[6]

            # Convert to boolean mask
            # Create empty mask with same dimensions as RGB image
            mask_bool = self.mask > 0  # True where mask is white

            rgb_masked = np.where(mask_bool[:, :, None], self.raw_rgb, 0).astype(np.uint8)
            depth_masked = np.where(mask_bool, self.raw_depth, 0)
            
            # Ensure depth is in uint16 (mm)
            if depth_masked.dtype != np.uint16:
                depth_masked = (depth_masked * 1000).astype(np.uint16)
            self.masked_depth = depth_masked
            
            # Create Open3D RGBD image
            rgb_o3d = o3d.geometry.Image(rgb_masked)
            depth_o3d = o3d.geometry.Image(depth_masked)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color=rgb_o3d,
                depth=depth_o3d,
                depth_scale=1000.0,
                depth_trunc=3.0,
                convert_rgb_to_intensity=False
            )

            intrinsic = o3d.camera.PinholeCameraIntrinsic(width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy)
            
            # Generate Point Cloud
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

            pcd = self.removeOutliers(pcd)

            return pcd
            
        except Exception as e:
            print(f"[covertToPCD] Error: {e}")
            return None

    def defineSegments(
        self,
        pcd,
        min_sample_pairwise_dist=0.003,
        max_sample_normal_angle_deg=40.0,
        max_inlier_normal_angle_deg=60.0,
        depth_tolerance=0.08,
        max_cluster_distance=0.02,
    ):

        try:
            start_time = time.time()

            # Downsample for thresholds and logging only
            voxel_size = 0.01
            downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)

            original_points = len(pcd.points)
            downsampled_points = len(downsampled.points)

            remaining_points_threshold = downsampled_points // self.segmentation_threshold
            cloud_segments = []
            remaining = copy.deepcopy(pcd)

            max_segments = 20
            min_inliers = 20
            segment_count = 0

            # RANSAC parameters
            ransac_n = 3
            num_iterations = 30
            distance_threshold = self.distance_thresh

            # Default cluster distance if not supplied
            if max_cluster_distance is None:
                # a reasonable heuristic: around three voxels or three times distance threshold
                max_cluster_distance = max(3.0 * distance_threshold,
                                           3.0 * voxel_size)

            # If not provided, default to previous heuristic
            if min_sample_pairwise_dist is None:
                min_sample_pairwise_dist = 5.0 * distance_threshold

            # Precompute cosines for normal angle thresholds
            cos_sample_normal_thresh = np.cos(
                np.deg2rad(max_sample_normal_angle_deg)
            )
            cos_inlier_normal_thresh = np.cos(
                np.deg2rad(max_inlier_normal_angle_deg)
            )

            while (len(remaining.points) > remaining_points_threshold and
                   segment_count < max_segments):

                if len(remaining.points) < min_inliers:
                    break

                # Ensure normals are available
                if not remaining.has_normals():
                    remaining.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamHybrid(
                            radius=0.05, max_nn=30
                        )
                    )

                points = np.asarray(remaining.points)
                normals = np.asarray(remaining.normals)

                # Depth assumed to be z coordinate
                depths = points[:, 2]
                depth_range = depths.max() - depths.min() if len(depths) > 0 else 0.0

                # If depth_tolerance not given, use heuristic
                if depth_tolerance is None:
                    current_depth_tolerance = max(
                        0.02, 0.1 * depth_range + 2.0 * distance_threshold
                    )
                else:
                    current_depth_tolerance = depth_tolerance

                best_inliers_idx = None
                best_inlier_count = 0

                N = points.shape[0]
                if N < ransac_n:
                    break

                # Custom RANSAC loop
                for _ in range(num_iterations):
                    max_sampling_tries = 50
                    idx_triplet = None

                    for _try in range(max_sampling_tries):
                        candidate_idx = np.random.choice(N, size=ransac_n, replace=False)
                        p1, p2, p3 = points[candidate_idx]

                        # Pairwise distances between sampled points
                        d12 = np.linalg.norm(p2 - p1)
                        d13 = np.linalg.norm(p3 - p1)
                        d23 = np.linalg.norm(p3 - p2)
                        if min(d12, d13, d23) < min_sample_pairwise_dist:
                            continue  # too close together

                        # Normal consistency of sampled points
                        n1, n2, n3 = normals[candidate_idx]
                        n1n2 = np.abs(np.dot(n1, n2))
                        n1n3 = np.abs(np.dot(n1, n3))
                        n2n3 = np.abs(np.dot(n2, n3))
                        if (n1n2 < cos_sample_normal_thresh or
                            n1n3 < cos_sample_normal_thresh or
                            n2n3 < cos_sample_normal_thresh):
                            continue

                        # Depth similarity between sampled points
                        z1, z2, z3 = depths[candidate_idx]
                        if (abs(z1 - z2) > current_depth_tolerance or
                            abs(z1 - z3) > current_depth_tolerance or
                            abs(z2 - z3) > current_depth_tolerance):
                            continue

                        idx_triplet = candidate_idx
                        break

                    if idx_triplet is None:
                        # Failed to find a suitable triplet this iteration
                        continue

                    p1, p2, p3 = points[idx_triplet]

                    # Fit plane from three points
                    v1 = p2 - p1
                    v2 = p3 - p1
                    n = np.cross(v1, v2)
                    norm_n = np.linalg.norm(n)
                    if norm_n < 1e-6:
                        continue  # degenerate

                    n = n / norm_n
                    d = -np.dot(n, p1)

                    # Point to plane distances
                    distances = np.abs(points @ n + d)

                    # Normal consistency for inliers
                    normal_cos = np.abs(normals @ n)

                    # Depth consistency relative to seed mean depth
                    mean_depth_seed = depths[idx_triplet].mean()
                    depth_diff = np.abs(depths - mean_depth_seed)

                    inlier_mask = (
                        (distances < distance_threshold) &
                        (normal_cos > cos_inlier_normal_thresh) &
                        (depth_diff < current_depth_tolerance)
                    )

                    inlier_idx = np.nonzero(inlier_mask)[0]
                    inlier_count = inlier_idx.size

                    if inlier_count > best_inlier_count:
                        best_inlier_count = inlier_count
                        best_inliers_idx = inlier_idx

                # Early exit conditions
                if best_inliers_idx is None or best_inlier_count < min_inliers:
                    break

                # Construct inlier cloud for this plane
                inlier_cloud = remaining.select_by_index(best_inliers_idx.tolist())

                # NEW: split this plane into spatially connected clusters
                labels = np.array(
                    inlier_cloud.cluster_dbscan(
                        eps=max_cluster_distance,
                        min_points=min_inliers,
                        print_progress=False
                    )
                )

                if labels.size == 0 or labels.max() < 0:
                    # no valid clustering, treat as a single region
                    cloud_segments.append(inlier_cloud)
                    segment_count += 1
                else:
                    for lbl in range(labels.max() + 1):
                        cluster_idx = np.where(labels == lbl)[0]
                        if cluster_idx.size < min_inliers:
                            continue
                        cluster_cloud = inlier_cloud.select_by_index(
                            cluster_idx.tolist()
                        )
                        cloud_segments.append(cluster_cloud)
                        segment_count += 1
                        if segment_count >= max_segments:
                            break

                # Remove all inliers for this plane from the remaining cloud
                remaining = remaining.select_by_index(
                    best_inliers_idx.tolist(), invert=True
                )

                if len(remaining.points) < min_inliers * 2:
                    break
                if segment_count >= max_segments:
                    break

            if self.debug:
                elapsed = time.time() - start_time
                print(
                    f"pointCloudData: Downsampled {original_points} to {downsampled_points} points. "
                    f"Segmented into {len(cloud_segments)} regions in {elapsed:.3f}s"
                )

            return cloud_segments

        except Exception as e:
            print(f"pointCloudData [defineSegments] Error: {e}")
            return None


    def getPCD(self):
        return self.pcd

    def getCenter(self):
        return self.pcd.get_center()

    def getCloudSegments(self):
        return self.defineSegments(self.pcd)