import numpy as np
import open3d as o3d
import cv2
import json
from typing import Literal
import os

class PointCloudData:

    def __init__(self, object_ID, input_type: Literal['RGB and DEPTH', 'RGBD', 'POINT CLOUD DATA'], raw_data_1, raw_depth=None, raw_mask=None, camera_info=None):
        self.object_ID = object_ID
        self.input_type = input_type

        self.raw_data_1 = raw_data_1[0]  # rgb, rgbd, or pcd
        self.raw_depth = raw_depth[0] if raw_depth else None
        self.raw_mask = raw_mask[0] if raw_mask else None
        self.camera_info = camera_info

        self.print = lambda *args, **kwargs: print("Point Cloud Data:", *args, **kwargs)

        self.pcd = self.covertToPCD()

        if self.pcd is not None and len(self.pcd.points) > 0:
            self.centroid = self.findCentroid()
            self.boundingBox = self.findBoundingBox()
            self.axis = self.findAxis()
        else:
            self.centroid = None
            self.boundingBox = None
            self.print("Warning: Empty point cloud. Centroid and bounding box not computed.")

    def covertToPCD(self):
        if self.input_type == 'RGB and DEPTH':
            assert os.path.exists(self.raw_data_1), f"RGB image not found: {self.raw_data_1}"
            assert os.path.exists(self.raw_depth), f"Depth image not found: {self.raw_depth}"
            assert os.path.exists(self.raw_mask), f"Mask image not found: {self.raw_mask}"
            assert os.path.exists(self.camera_info), f"Camera info not found: {self.camera_info}"

            rgb = cv2.cvtColor(cv2.imread(self.raw_data_1), cv2.COLOR_BGR2RGB)
            depth = cv2.imread(self.raw_depth, cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(self.raw_mask, cv2.IMREAD_GRAYSCALE)
        
            with open(self.camera_info, 'r') as f:
                scene_info = json.load(f)["0"]
                K = np.array(scene_info["cam_K"]).reshape(3, 3)
                depth_scale = float(scene_info.get("depth_scale", 1.0))

            if mask.shape != depth.shape:
                self.print("Resizing mask to match depth resolution...")
                mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)

            if depth.dtype == np.float32 or depth.dtype == np.float64:
                self.print("Detected depth in metres, setting depth_scale = 1.0")
                depth_scale = 1.0
            elif depth.dtype == np.uint16 or depth.dtype == np.uint32:
                self.print("Assuming depth in millimetres, setting depth_scale = 1000.0")
                depth_scale = 1000.0
            else:
                raise ValueError(f"Unsupported depth dtype: {depth.dtype}")

            self.print("Depth min:", np.min(depth), "max:", np.max(depth))

            valid_mask = (mask > 0) & (depth > 0)
            self.print("Valid points in mask:", np.count_nonzero(valid_mask))

            if np.count_nonzero(valid_mask) == 0:
                self.print("No valid masked points found — check depth or mask alignment.")
                return None

            depth_masked = np.where(valid_mask, depth, 0)
            rgb_masked = np.zeros_like(rgb)
            rgb_masked[valid_mask] = rgb[valid_mask]

            depth_o3d = o3d.geometry.Image(depth_masked.astype(np.uint16))
            color_o3d = o3d.geometry.Image(rgb_masked.astype(np.uint8))
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d,
                depth_scale=depth_scale,
                depth_trunc=3.0,
                convert_rgb_to_intensity=False
            )

            h, w = depth.shape
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
            pcd.transform([[1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]])
            
            return pcd

        elif self.input_type == 'RGBD':
            return o3d.geometry.PointCloud.create_from_rgbd_image(self.raw_data_1)

        elif self.input_type == 'POINT CLOUD DATA':
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.raw_data_1)
            return pcd

        else:
            self.print(f"Invalid input_type: {self.input_type}")
            return None

    def findBoundingBox(self):
        return self.pcd.get_axis_aligned_bounding_box()

    def findCentroid(self):
        return np.mean(np.asarray(self.pcd.points), axis=0)
    
    def findAxis(self):
        if self.pcd is None or len(self.pcd.points) == 0:
            self.print("Cannot compute axis: Point cloud is empty.")
            return None

        points = np.asarray(self.pcd.points)
        centroid = self.centroid

        # Center the points
        centered_points = points - centroid

        # Compute covariance matrix
        cov_matrix = np.cov(centered_points, rowvar=False)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvectors by descending eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        principal_direction = eigenvectors[:, idx[0]]  # First principal component

        # Return a line representation: origin (centroid) and direction vector
        return {
            "origin": centroid,
            "direction": principal_direction
        } 

    def getPCD(self):
        return self.pcd

    def getCentroid(self):
        return self.centroid
    
    def getBoundingBox(self):
        return self.boundingBox
    
    def getAxis(self):
        return self.axis

    def update(self):
        pass
