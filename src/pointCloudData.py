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

        
    def defineSegments(self, pcd):
        """
            Segments the point cloud into multiple planes using RANSAC
        """     
        try:   
            start_time = time.time()

            remaing_points_threshold = len(pcd.points) // self.segmentation_threshold
            cloud_segments = []
            remaining = copy.deepcopy(pcd)

            while len(remaining.points) > remaing_points_threshold:
                _, inliers = remaining.segment_plane(distance_threshold=self.distance_thresh,
                                                    ransac_n=3,
                                                    num_iterations=1000,
                                                    probability=0.999)
                
                inlier_cloud = remaining.select_by_index(inliers)

                cloud_segments.append(inlier_cloud)
                remaining = remaining.select_by_index(inliers, invert=True)

            if self.debug:
                print(f"pointCloudData: Segmented into {len(cloud_segments)} planes. Generated within {time.time() - start_time:.3f}s")

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