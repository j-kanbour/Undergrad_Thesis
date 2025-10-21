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
from cv_bridge import CvBridge

class PointCloudData:

    def __init__(self, raw_rgb, raw_depth, mask, camera_info, nearest_neighbor=500, distance_thresh=0.005, semght_threshold=10, debug=False):

        #convert ROS images to CV2
        self.bridge = CvBridge()
        self.raw_rgb = self.bridge.imgmsg_to_cv2(raw_rgb, 'bgr8')
        self.raw_depth = self.bridge.imgmsg_to_cv2(raw_depth, desired_encoding="passthrough")

        self.debug = debug
        self.mask = mask
        self.camera_info = camera_info
        self.nearest_neighbout = nearest_neighbor
        self.distance_thresh = distance_thresh
        self.segmentation_threshold = semght_threshold


        start_time = time.time()

        #convert extracted data to point cloud
        self.pcd = self.covertToPCD()

        if self.debug:
            print(f"pointCloudData: Point cloud generated within {time.time() - start_time:.3f}s")
    
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
        """
            Applied mask to masked RGB-D image and generated a point cloud object from the results
        """

        try:
            # Apply mask to RGB and Depth images
            mask = np.zeros((self.camera_info.height, self.camera_info.width), dtype=np.uint8)
            points = np.array(self.mask).reshape(-1, 2)
            mask = cv2.fillPoly(mask, [points.astype(np.int32)], 255)
            mask = mask.astype(bool) 
            rgb_masked = np.where(mask[:, :, None], self.raw_rgb, 0).astype(np.uint8)
            depth_masked = np.where(mask, self.raw_depth, 0)
            
            # Ensure depth is in uint16 (mm)
            if depth_masked.dtype != np.uint16:
                depth_masked = (depth_masked * 1000).astype(np.uint16)
            
            # Create Open3D RGBD image
            rgb_o3d = o3d.geometry.Image(rgb_masked)
            depth_o3d = o3d.geometry.Image(depth_masked)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color=rgb_o3d,
                depth=depth_o3d,
                depth_scale=1000.0,
                depth_trunc=10,
                convert_rgb_to_intensity=False
            )

            # Extract Camera intrinsics
            K = np.array(self.camera_info.K).reshape(3, 3)
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]
            w = self.camera_info.width
            h = self.camera_info.height
            intrinsic = o3d.camera.PinholeCameraIntrinsic(width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy)
            
            # Generate Point Cloud with open3d
            pointCloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

            if self.debug:
                print(f'pointCloudData: pointcloud size', len(pointCloud.points))

            # remove outliers based on nearest neighbour
            pointCloud = self.removeOutliers(pointCloud)

            return pointCloud
            
        except Exception as e:
            print(f"pointCloudData [covertToPCD] Error: {e}")
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