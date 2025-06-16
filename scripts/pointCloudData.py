import numpy as np
import open3d as o3d
import cv2
import json
from typing import Literal
import os

class PointCloudData:

    def __init__(self, object_ID, input_type: Literal['RGB and DEPTH', 'RGBD', 'RGBD STREAM'], raw_data_1, raw_depth=None, raw_mask=None, camera_info=None):
        self.print = lambda *args, **kwargs: print("Point Cloud Data:", *args, **kwargs)

        self.object_ID = object_ID
        self.input_type = input_type

        self.pcd = self.covertToPCD(raw_data_1[0], raw_depth[0], raw_mask[0], camera_info)

        if self.pcd and len(self.pcd.points) > 0:
            self.centroid = self.findCentroid()
            self.boundingBox = self.findBoundingBox()
            self.axis = self.findAxis()
        else:
            self.centroid = None
            self.boundingBox = None
            self.print("Warning: Empty point cloud. Centroid and bounding box not computed.")

    def removeOutliers(self, pcd):
        """
            Remotes outliers based on nerious neighbour algorithm
            NOTE: Increasing the effect of this increases run time
        """

        if pcd.is_empty():
            print("Warning: Provided point cloud is empty.")
            return pcd

        # Efficient parameters
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=500, std_ratio=0.25)
        return pcd.select_by_index(ind)
        
    def _covertToPCD_helper_(self, rgb, depth, mask, K, depth_scale):
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
        rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=depth_scale,
            depth_trunc=3.0,
            convert_rgb_to_intensity=False
        )

        h, w = depth.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_o3d, intrinsic)
        pcd.transform([[1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1]])

        return self.removeOutliers(pcd)


    def covertToPCD(self, raw_data_1, raw_depth, raw_mask, camera_info):
        """
            Build Point Cloud Data from various input types
            Input types include: 
                - RGB and DEPTH
                - RGBD
                - RGBD STREAM
        """

        if self.input_type == 'RGB and DEPTH':
            assert os.path.exists(raw_data_1), f"RGB image not found: {raw_data_1}"
            assert os.path.exists(raw_depth), f"Depth image not found: {raw_depth}"
            assert os.path.exists(raw_mask), f"Mask image not found: {raw_mask}"
            assert os.path.exists(camera_info), f"Camera info not found: {camera_info}"

            rgb = cv2.cvtColor(cv2.imread(raw_data_1), cv2.COLOR_BGR2RGB)
            depth = cv2.imread(raw_depth, cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(raw_mask, cv2.IMREAD_GRAYSCALE)

            with open(camera_info, 'r') as f:
                scene_info = json.load(f)["0"]
                K = np.array(scene_info["cam_K"]).reshape(3, 3)
                depth_scale = float(scene_info.get("depth_scale", 1.0))

            return self._build_pcd_from_rgb_depth_mask(rgb, depth, mask, K, depth_scale)

        elif self.input_type == 'RGBD':
            rgbd = np.load(raw_data_1) if raw_data_1.endswith(".npy") else cv2.imread(raw_data_1, cv2.IMREAD_UNCHANGED)
            if rgbd.ndim != 3 or rgbd.shape[2] < 4:
                self.print("Error: Expected RGBD image with 4 channels (RGB + Depth)")
                return None

            rgb = rgbd[:, :, :3]
            depth = rgbd[:, :, 3]
            mask = cv2.imread(raw_mask, cv2.IMREAD_GRAYSCALE)

            with open(camera_info, 'r') as f:
                scene_info = json.load(f)["0"]
                K = np.array(scene_info["cam_K"]).reshape(3, 3)
                depth_scale = float(scene_info.get("depth_scale", 1.0))

            return self._build_pcd_from_rgb_depth_mask(rgb, depth, mask, K, depth_scale)

        elif self.input_type == 'RGBD STREAM':
            rgbd = raw_data_1
            mask = raw_mask
            if rgbd.ndim != 3 or rgbd.shape[2] < 4:
                self.print("Error: Expected RGBD stream with 4 channels (RGB + Depth)")
                return None

            rgb = rgbd[:, :, :3]
            depth = rgbd[:, :, 3]
            K = np.array(camera_info["cam_K"]).reshape(3, 3)
            depth_scale = float(camera_info.get("depth_scale", 1.0))

            return self._build_pcd_from_rgb_depth_mask(rgb, depth, mask, K, depth_scale)

        else:
            self.print(f"Invalid input_type: {self.input_type}")
            return None

    def findBoundingBox(self):
        """
            Build 3D bounding box around 
            - May want to orientate this around the Axis
        """
        return self.pcd.get_oriented_bounding_box(True)

    def findCentroid(self):

        # return np.mean(np.asarray(self.pcd.points), axis=0)
        return self.pcd.get_center()

    def findAxis(self):
        return self.boundingBox.R

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
