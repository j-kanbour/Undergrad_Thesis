#!/usr/bin/env python3.8
import numpy as np
import cv2
import open3d as o3d
import json
import os
import copy
import matplotlib.pyplot as plt

class PointCloudData:

    def __init__(self, object_ID, raw_rgb, raw_depth, mask, camera_info):
        self.print = lambda *args, **kwargs: print("Point Cloud Data:", *args, **kwargs)
        assert os.path.exists(raw_rgb), f"RGB image not found: {raw_rgb}"
        assert os.path.exists(raw_depth), f"Depth image not found: {raw_depth}"
        assert os.path.exists(mask), f"Mask image not found: {mask}"

        for path, name in [(raw_rgb, "RGB"), (raw_depth, "Depth"), (mask, "Mask")]:
            assert os.path.exists(path), f"{name} image not found: {path}"

        # Load images
        self.object_ID = object_ID
        self.raw_rgb = cv2.cvtColor(cv2.imread(raw_rgb), cv2.COLOR_BGR2RGB)
        self.raw_depth = cv2.imread(raw_depth, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        self.mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)


        self.masked_depth = None
        self.camera_info = self.extractCameraInfo(camera_info)

        #convert superquadric parameters to pcd
        self.pcd = self.covertToPCD()

        # self.centroid = self.findCentroid()
        # self.boundingBox = self.findBoundingBox()
        # self.axis = self.findAxis()

    def extractCameraInfo(self, camera_info):

        json_path, cam_id = camera_info
        with open(json_path, "r") as f:
            data = json.load(f)
        block = data[str(cam_id)]

        # Build outputs
        K = np.array(block["cam_K"], dtype=np.float64).reshape(3, 3)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        # Width/height: try to infer from loaded colour image on the class (if present)
        w = h = None
        if hasattr(self, "raw_rgb") and isinstance(self.raw_rgb, np.ndarray) and self.raw_rgb.ndim >= 2:
            h, w = self.raw_rgb.shape[:2]
        return [K, fx, fy, cx, cy, w, h]

    # #mirror the partial point cloud around center axis for more refined model
    # def mirror_cloud(self, pcd, keep_original=True):
        
    #     try:
    #         if pcd.is_empty():
    #             raise ValueError("Input point cloud is empty")

    #         # ── 1. Compute centroid and translate to local frame ───────────────
    #         pts = np.asarray(pcd.points)
    #         bbox = pcd.get_oriented_bounding_box(True)
    #         center = bbox.center

    #         pts_local = pts - center  # move centroid to origin

    #         # ── 2. Reflect across the origin (x,y,z → -x,-y,-z) ────────────────
    #         pts_mirror = -pts_local

    #         # ── 3. Bring mirrored points back to sensor/world frame ────────────
    #         pts_mirror_world = pts_mirror + center

    #         # ── 4. Build mirrored cloud, copying colours + normals if present ──
    #         mirrored = o3d.geometry.PointCloud()
    #         mirrored.points = o3d.utility.Vector3dVector(pts_mirror_world)

    #         # copy RGB colours if they exist
    #         if pcd.has_colors():
    #             colours = np.asarray(pcd.colors)
    #             mirrored.colors = o3d.utility.Vector3dVector(colours)

    #         # copy (and flip) normals if they exist
    #         if pcd.has_normals():
    #             normals = np.asarray(pcd.normals)
    #             mirrored.normals = o3d.utility.Vector3dVector(-normals)

    #         # ── 5. Combine or return only mirrored part ────────────────────────
    #         if keep_original:
    #             combined = o3d.geometry.PointCloud()
    #             combined += pcd
    #             combined += mirrored
    #             return combined
    #         else:
    #             return mirrored
    #     except Exception as e:
    #         print(f"Mirror Error: {e}")

    #remove outliers from, may not need if mask is good
    def removeOutliers(self, pcd):
        """
            Remotes outliers based on nerious neighbour algorithm
            NOTE: Increasing the effect of this increases run time
        """
        try:
            if pcd.is_empty():
                print("Warning: Provided point cloud is empty.")
                return pcd

            # Efficient parameters
            _, ind = pcd.remove_statistical_outlier(nb_neighbors=500, std_ratio=0.25)

            return pcd.select_by_index(ind)

        except Exception as e:
            print(f"[removeOutliers] Error: {e}")
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
            print(f'CHECKPOINT:{pcd}')
            pcd = self.removeOutliers(pcd)

            return pcd
            
        except Exception as e:
            print(f"[covertToPCD] Error: {e}")
            return None

    def defineSegments(self, pcd):
            
        remaing_points_threshold = len(pcd.points) // 10
        cloud_segments = []
        remaining = copy.deepcopy(pcd)
        colors = plt.cm.get_cmap("tab10", 10)
        count = 0
        while len(remaining.points) > remaing_points_threshold:
            _, inliers = remaining.segment_plane(distance_threshold=0.03,
                                                    ransac_n=3,
                                                    num_iterations=1000,
                                                    probability=0.999)
            # [a, b, c, d] = plane_model.tolist()
            # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

            inlier_cloud = remaining.select_by_index(inliers)
            #inlier_cloud = self.removeOutliers(inlier_cloud)
            inlier_cloud.paint_uniform_color(colors(count)[:3])

            cloud_segments.append(inlier_cloud)
            remaining = remaining.select_by_index(inliers, invert=True)

        return cloud_segments

    # def findBoundingBox(self):
    #     return self.pcd.get_oriented_bounding_box(True)

    # def findCentroid(self):
    #     return self.pcd.get_center()

    # def findAxis(self):
    #     return self.boundingBox.R

    def getPCD(self):
        return self.pcd

    # def getCentroid(self):
    #     return self.centroid

    # def getBoundingBox(self):
    #     return self.boundingBox

    # def getAxis(self):
    #     return self.axis
    
    def getCloudSegments(self):
        return self.defineSegments(self.pcd)
    
    def getRawData(self):
        return {
            "object_ID" : self.object_ID,
            "raw_rgb" : self.raw_rgb,
            "raw_depth" : self.raw_depth,
            "masked_depth" : self.masked_depth,
            "mask" : self.mask,
            "camera_info" : self.camera_info
        }