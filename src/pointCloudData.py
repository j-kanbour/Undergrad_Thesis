#!/usr/bin/env python3.8
import numpy as np
import open3d as o3d
import cv2

class PointCloudData:

    def __init__(self, object_ID, input_type, raw_data_1, bbox=None, raw_depth=None, raw_mask=None, camera_info=None):
        self.print = lambda *args, **kwargs: print("Point Cloud Data:", *args, **kwargs)

        self.object_ID = object_ID
        self.input_type = input_type
        self.bbox = bbox
        self.raw_data_1 = raw_data_1
        self.raw_depth = raw_depth
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
        
    # def _covertToPCD_helper_(self, rgb, depth, mask, K, depth_scale):
    #     if mask.shape != depth.shape:
    #         self.print("Resizing mask to match depth resolution...")
    #         mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)

    #     if depth.dtype == np.float32 or depth.dtype == np.float64:
    #         self.print("Detected depth in metres, setting depth_scale = 1.0")
    #         depth_scale = 1.0
    #     elif depth.dtype == np.uint16 or depth.dtype == np.uint32:
    #         self.print("Assuming depth in millimetres, setting depth_scale = 1000.0")
    #         depth_scale = 1000.0
    #     else:
    #         raise ValueError(f"Unsupported depth dtype: {depth.dtype}")

    #     self.print("Depth min:", np.min(depth), "max:", np.max(depth))

    #     valid_mask = (mask > 0) & (depth > 0)
    #     self.print("Valid points in mask:", np.count_nonzero(valid_mask))

    #     if np.count_nonzero(valid_mask) == 0:
    #         self.print("No valid masked points found — check depth or mask alignment.")
    #         return None

    #     depth_masked = np.where(valid_mask, depth, 0)
    #     rgb_masked = np.zeros_like(rgb)
    #     rgb_masked[valid_mask] = rgb[valid_mask]

    #     depth_o3d = o3d.geometry.Image(depth_masked.astype(np.uint16))
    #     color_o3d = o3d.geometry.Image(rgb_masked.astype(np.uint8))
    #     rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
    #         color_o3d, depth_o3d,
    #         depth_scale=depth_scale,
    #         depth_trunc=3.0,
    #         convert_rgb_to_intensity=False
    #     )

    #     h, w = depth.shape
    #     fx, fy = K[0, 0], K[1, 1]
    #     cx, cy = K[0, 2], K[1, 2]
    #     intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

    #     pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_o3d, intrinsic)
    #     pcd.transform([[1, 0, 0, 0],
    #                 [0, -1, 0, 0],
    #                 [0, 0, -1, 0],
    #                 [0, 0, 0, 1]])

    #     return self.removeOutliers(pcd)


    def covertToPCD(self):
        """
            Build Point Cloud Data from various input types
            Input types include: 
                - RGB and DEPTH
                - RGBD
                - RGBD STREAM #ROBOT IMPLEMENTATION
        """
        raw_data_1 = self.raw_data_1       # RGB image or RGBD image
        raw_depth = self.raw_depth         # Depth image or None
        raw_mask = self.raw_mask           # Binary mask or None
        bbox = self.bbox                   # Bounding box or None
        camera_info = self.camera_info     # sensor_msgs/CameraInfo

        if self.input_type == 'RGBD STREAM':
            # Step 1: Handle Mask
            mask = None
            if raw_mask is not None:
                mask = raw_mask.astype(bool)

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

            # Flip orientation: convert OpenCV camera to Open3D camera coordinates
            # pcd.transform([[1, 0, 0, 0],
            #             [0, -1, 0, 0],
            #             [0, 0, -1, 0],
            #             [0, 0, 0, 1]])

            # Step 5: Optional cleanup
            pcd = self.removeOutliers(pcd)

            return pcd
        
            """
            
            NOTE: Below implementations mainly for testing
            
            elif self.input_type == 'RGB and DEPTH':
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

                return self._covertToPCD_helper_(rgb, depth, mask, K, depth_scale)

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

                return self._covertToPCD_helper_(rgb, depth, mask, K, depth_scale)
            """

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
    
    def getRawData(self):
        return {
            "object_ID" : self.object_ID,
            "input_type" : self.input_type,
            "raw_data_1" : self.raw_data_1,
            "raw_depth" : self.raw_depth,
            "raw_mask" : self.raw_mask,
            "camera_info" : self.camera_info
        }

    def update(self):
        pass


"""

    TODO 

    Need to integrate with ROS and robot to handle data streaming
    
    - Will need to change how data is handled and red by all other components (grasps, superquadrics)
    - Fing way to remove the need for generating point cloud object and making claculation straight from depth image

"""