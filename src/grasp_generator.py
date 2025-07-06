#!/usr/bin/env python3.8

import rospy
import numpy as np
import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from unsw_vision_msgs.msg import DetectionList
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

import open3d as o3d
from pointCloudData import PointCloudData
from fit_superquadric import fit_superquadric_cloud


def o3d_to_ros(pcd: o3d.geometry.PointCloud, frame_id: str) -> PointCloud2:
    """Convert Open3D point cloud → ROS PointCloud2 (XYZ only)."""
    pts = np.asarray(pcd.points, dtype=np.float32)
    if pts.size == 0:
        return None

    header = rospy.Header(frame_id=frame_id, stamp=rospy.Time.now())
    fields = [
        PointField('x', 0,  PointField.FLOAT32, 1),
        PointField('y', 4,  PointField.FLOAT32, 1),
        PointField('z', 8,  PointField.FLOAT32, 1)
    ]
    return pc2.create_cloud(header, fields, pts)


class SQPublisher:
    """ROS node that publishes a fitted SQ mesh for each detected object."""

    def __init__(self):
        rospy.init_node('sq_fitter_node')
        self.bridge = CvBridge()

        # Topics ------------------------------------------------------------
        rgb   = '/hsrb/head_rgbd_sensor/rgb/image_raw'
        depth = '/hsrb/head_rgbd_sensor/depth_registered/image_raw'
        info  = '/hsrb/head_rgbd_sensor/depth_registered/camera_info'
        det   = '/unsw_vision/detections/objects/positions'

        self.rgb_sub   = message_filters.Subscriber(rgb,   Image)
        self.depth_sub = message_filters.Subscriber(depth, Image)
        self.sync      = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=10, slop=0.05)
        self.sync.registerCallback(self.rgbd_cb)

        self.info_sub  = rospy.Subscriber(info, CameraInfo, self.info_cb)
        self.det_sub   = rospy.Subscriber(det,  DetectionList, self.det_cb)

        self.cloud_pub = rospy.Publisher('/sq_fitted_cloud', PointCloud2, queue_size=10)

        self.latest_info = None
        self.latest_dets = None
        self.last_det_ts = rospy.Time(0)
        rospy.loginfo('sq_fitter_node up and running.')

    # ------------------------------------------------------------------
    def info_cb(self, msg):
        self.latest_info = msg

    def det_cb(self, msg):
        self.latest_dets = msg
        self.last_det_ts = rospy.Time.now()

    # ------------------------------------------------------------------
    def rgbd_cb(self, rgb_msg: Image, depth_msg: Image):
        if self.latest_dets is None or (rospy.Time.now() - self.last_det_ts).to_sec() > 2.0:
            rospy.loginfo_throttle(2.0, 'No detections – skipping frame.')
            return

        header_frame = rgb_msg.header.frame_id
        published_any = False

        for obj in self.latest_dets.objects:
            if obj.object_class not in {'cup', 'can', 'bottle'}:
                continue

            # Build point cloud for this object --------------------------------
            pcd_builder = PointCloudData(
                object_ID=obj.tracking_id,
                raw_data_1=rgb_msg,
                bbox=obj.bbox,
                raw_depth=depth_msg,
                raw_mask=None,
                camera_info=self.latest_info
            )
            pcd = pcd_builder.getPCD()
            if pcd is None or pcd.is_empty():
                rospy.logwarn(f'ID {obj.tracking_id}: empty PCD – skipping.')
                continue

            # Fit superquadric -------------------------------------------------
            try:
                sq_cloud, _ = fit_superquadric_cloud(
                                    pcd,
                                    class_name=obj.object_class,        # pass detection label here
                                    mirror_first=True)
                #rospy.loginfo(f'ID {obj.tracking_id}: SQ fitted, RMS={rms:.3e}')
            except Exception as exc:
                rospy.logwarn(f'ID {obj.tracking_id}: fit failed – {exc}')
                continue

            # Publish ---------------------------------------------------------
            ros_cloud = o3d_to_ros(sq_cloud, frame_id=header_frame)
            if ros_cloud:
                self.cloud_pub.publish(ros_cloud)
                published_any = True

        if not published_any:
            rospy.loginfo_throttle(2.0, 'No SQ clouds published this frame.')


if __name__ == '__main__':
    try:
        node = SQPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
