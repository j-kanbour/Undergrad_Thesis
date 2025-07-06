#!/usr/bin/env python3.8
import numpy as np
import open3d as o3d
import cv2
from cv_bridge import CvBridge

class PointCloudData:

    def __init__(self, object_ID, raw_data_1, bbox=None, raw_depth=None, raw_mask=None, camera_info=None):
        self.print = lambda *args, **kwargs: print("Point Cloud Data:", *args, **kwargs)
        self.bridge = CvBridge()

        self.object_ID = object_ID
        self.bbox = bbox
        self.raw_data_1 = self.bridge.imgmsg_to_cv2(raw_data_1, 'bgr8')
        self.raw_depth = self.bridge.imgmsg_to_cv2(raw_depth, desired_encoding="passthrough")
        self.raw_mask = raw_mask
        self.camera_info = camera_info

        self.pcd = self.covertToPCD()

        if self.pcd and len(self.pcd.points) > 0:
            self.centroid = self.findCentroid()
            self.boundingBox = self.findBoundingBox()
            self.axis = self.findAxis()
        else:
            self.centroid = None
            self.boundingBox = None
            self.print("Warning: Empty point cloud. Centroid and bounding box not computed.")

    def mirror_cloud(self, pcd, keep_original=False):
        if pcd.is_empty():
            raise ValueError("Input point cloud is empty")

        # ── 1. Compute centroid and translate to local frame ───────────────
        pts = np.asarray(pcd.points)
        centre = pts.mean(axis=0)

        pts_local = pts - centre  # move centroid to origin

        # ── 2. Reflect across the origin (x,y,z → -x,-y,-z) ────────────────
        pts_mirror = -pts_local

        # ── 3. Bring mirrored points back to sensor/world frame ────────────
        pts_mirror_world = pts_mirror + centre

        # ── 4. Build mirrored cloud, copying colours + normals if present ──
        mirrored = o3d.geometry.PointCloud()
        mirrored.points = o3d.utility.Vector3dVector(pts_mirror_world)

        # copy RGB colours if they exist
        if pcd.has_colors():
            colours = np.asarray(pcd.colors)
            mirrored.colors = o3d.utility.Vector3dVector(colours)

        # copy (and flip) normals if they exist
        if pcd.has_normals():
            normals = np.asarray(pcd.normals)
            mirrored.normals = o3d.utility.Vector3dVector(-normals)

        # ── 5. Combine or return only mirrored part ────────────────────────
        if keep_original:
            combined = o3d.geometry.PointCloud()
            combined += pcd
            combined += mirrored
            return combined
        else:
            return mirrored


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
            raw_data_1 = self.raw_data_1       # RGB image or RGBD image
            raw_depth = self.raw_depth         # Depth image or None
            bbox = self.bbox                   # Bounding box or None
            camera_info = self.camera_info     # sensor_msgs/CameraInfo

            # Step 1: Handle Mask
            """
            NOTE: Temporary until i get mask
            """
            mask = None
            if self.raw_mask is not None:
                mask = self.raw_mask.astype(bool)

            elif bbox is not None:

                #NOTE: tmp mask generation for until I get a mask from vision 

                # [x, y, w, h] = bbox → generate mask using OpenCV GrabCut
                x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height

                # Initialize mask for GrabCut
                grabcut_mask = np.zeros(raw_data_1.shape[:2], np.uint8)

                # Define rectangle for GrabCut
                rect = (x, y, w, h)

                # Allocate background and foreground models
                bg_model = np.zeros((1, 65), np.float64)
                fg_model = np.zeros((1, 65), np.float64)

                # Run GrabCut algorithm
                try:
                    cv2.grabCut(raw_data_1, grabcut_mask, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)

                    # Extract foreground and probable foreground as final mask
                    mask = np.where((grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 1, 0).astype(bool)

                    if np.count_nonzero(mask) < 50:  # Too few points, fallback
                        self.print("GrabCut mask too small. Falling back to rectangular mask.")
                        mask = np.zeros(raw_data_1.shape[:2], dtype=bool)
                        mask[y:y+h, x:x+w] = True

                    else:
                        self.print("Mask estimated using GrabCut from bounding box.")

                except Exception as e:
                    self.print(f"GrabCut failed: {e}. Falling back to rectangular mask.")
                    mask = np.zeros(raw_data_1.shape[:2], dtype=bool)
                    mask[y:y+h, x:x+w] = True
                    
            else:
                # Default: use full image
                mask = np.ones(raw_data_1.shape[:2], dtype=bool)

            self.raw_mask = mask

            # Step 2: RGB and Depth Handling
            if raw_depth is not None:
                # Apply mask
                rgb_masked = np.where(mask[:, :, None], raw_data_1, 0).astype(np.uint8)
                depth_masked = np.where(mask, raw_depth, 0)

                # Ensure depth format is uint16 (in mm)
                if depth_masked.dtype != np.uint16:
                    depth_masked = (depth_masked * 1000).astype(np.uint16)  # convert from meters to mm

                # Create Open3D RGBD image
                rgb_o3d = o3d.geometry.Image(rgb_masked)
                depth_o3d = o3d.geometry.Image(depth_masked)

                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color=rgb_o3d,
                    depth=depth_o3d,
                    depth_scale=1000.0,  # depth in mm
                    depth_trunc=3.0,
                    convert_rgb_to_intensity=False
                )

            else:
                self.print("No separate depth input found. Assuming raw_data_1 is already RGB-D.")
                # If depth is encoded in 4th channel
                if raw_data_1.shape[2] == 4:
                    rgb_masked = np.where(mask[:, :, None], raw_data_1[:, :, :3], 0).astype(np.uint8)
                    depth_masked = np.where(mask, raw_data_1[:, :, 3], 0).astype(np.uint16)

                    rgb_o3d = o3d.geometry.Image(rgb_masked)
                    depth_o3d = o3d.geometry.Image(depth_masked)

                    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                        color=rgb_o3d,
                        depth=depth_o3d,
                        depth_scale=1000.0,
                        depth_trunc=3.0,
                        convert_rgb_to_intensity=False
                    )
                else:
                    raise ValueError("raw_data_1 does not contain depth and raw_depth is not provided.")

            # Step 3: Camera intrinsics
            K = np.array(camera_info.K).reshape(3, 3)
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]
            w = camera_info.width
            h = camera_info.height

            intrinsic = o3d.camera.PinholeCameraIntrinsic(width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy)

            # Step 4: Generate Point Cloud
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

            # Step 5: Optional cleanup
            pcd = self.removeOutliers(pcd)

            
            """
            mirror partial point cloud 
            NOTE: Mirror is off centered as center is calculated as mean of partial pcd
            """
            #pcd = self.mirror_cloud(pcd)

            return pcd
    
        except Exception as e:
            print(f"[covertToPCD] Error: {e}")
            return None

    def findBoundingBox(self):
        return self.pcd.get_oriented_bounding_box(True)

    def findCentroid(self):
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
    
    def getRawData(self):
        return {
            "object_ID" : self.object_ID,
            "raw_data_1" : self.raw_data_1,
            "raw_depth" : self.raw_depth,
            "raw_mask" : self.raw_mask,
            "camera_info" : self.camera_info
        }