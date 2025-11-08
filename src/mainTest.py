
from pointCloudData import PointCloudData
from superquadric import Superquadric
from grasps import Grasps
import open3d as o3d
import sys, os, time
import psutil
import numpy as np
import copy
import cv2
import json
import psutil
import os
import time
from contextlib import contextmanager
import psutil
import os
import time
import pandas as pd
from contextlib import contextmanager

EXCEL_FILE = "perfromance.xlsx"

@contextmanager
def profile_block(name="Block"):
    process = psutil.Process(os.getpid())
    
    # Record initial stats
    start_cpu = process.cpu_times()
    start_mem = process.memory_info().rss / (1024 ** 2)  # MB
    start_time = time.time()

    yield  # Run the code inside the block

    # Record final stats
    end_cpu = process.cpu_times()
    end_mem = process.memory_info().rss / (1024 ** 2)
    end_time = time.time()

    # Compute differences
    user_cpu = end_cpu.user - start_cpu.user
    system_cpu = end_cpu.system - start_cpu.system
    mem_used = end_mem - start_mem
    elapsed_time = end_time - start_time

    # Prepare data to save
    data = {
        "Block": [name],
        "Time_s": [elapsed_time],
        "Memory_MB": [mem_used],
        "CPU_User_s": [user_cpu],
        "CPU_System_s": [system_cpu],
        "Timestamp": [time.strftime("%Y-%m-%d %H:%M:%S")]
    }

    df = pd.DataFrame(data)

    # Append to Excel file
    try:
        existing_df = pd.read_excel(EXCEL_FILE)
        df = pd.concat([existing_df, df], ignore_index=True)
    except FileNotFoundError:
        pass  # Excel file doesn't exist yet

    df.to_excel(EXCEL_FILE, index=False)

    # Optional: print summary
    print(f"\n[{name}] Performance Report saved to {EXCEL_FILE}")
    print(f"  ⏱️  Time elapsed: {elapsed_time:.3f} s")
    print(f"  🧠 Memory change: {mem_used:.3f} MB")
    print(f"  💻 CPU time (user): {user_cpu:.3f} s")
    print(f"  💻 CPU time (system): {system_cpu:.3f} s\n")



models = { 
            "1": {
                "rgb_path":"../data/rgb_and_depth_data/000001/rgb/000000.png",
                "depth_path":"../data/rgb_and_depth_data/000001/depth/000000.png",
                "mask_path":"../data/rgb_and_depth_data/000001/mask_visib/000000_000000.png",
                "scene_info_json":"../data/rgb_and_depth_data/000001/scene_camera.json",
                "class_name":"bottle",
                "object_ID":1
            },
            "2": {                
                "rgb_path":"../data/rgb_and_depth_data/000001/rgb/000001.png",
                "depth_path":"../data/rgb_and_depth_data/000001/depth/000002.png",
                "mask_path":"../data/rgb_and_depth_data/000001/mask_visib/000000_000001.png",
                "scene_info_json":"../data/rgb_and_depth_data/000001/scene_camera.json",
                "class_name":"can",
                "object_ID":2
            },
            "3": {
                "rgb_path":"../data/rgb_and_depth_data/000008/rgb/000000.png",
                "depth_path":"../data/rgb_and_depth_data/000008/depth/000000.png",
                "mask_path":"../data/rgb_and_depth_data/000008/mask_visib/000001_000001.png",
                "scene_info_json":"../data/rgb_and_depth_data/000008/scene_camera.json",
                "class_name":"can",
                "object_ID":3
            },
            "4": {                
                "rgb_path":"../data/rgb_and_depth_data/000005/rgb/000000.png",
                "depth_path":"../data/rgb_and_depth_data/000005/depth/000000.png",
                "mask_path":"../data/rgb_and_depth_data/000005/mask_visib/000001_000001.png",
                "scene_info_json":"../data/rgb_and_depth_data/000005/scene_camera.json",
                "class_name":"box",
                "object_ID":4
            },
            "5": {                
                "rgb_path":"../data/rgb_and_depth_data/000005/rgb/000004.png",
                "depth_path":"../data/rgb_and_depth_data/000005/depth/000004.png",
                "mask_path":"../data/rgb_and_depth_data/000005/mask_visib/000004_000010.png",
                "scene_info_json":"../data/rgb_and_depth_data/000005/scene_camera.json",
                "class_name":"bottle",
                "object_ID":5
            }
        }

""" Test 1: modelling"""
def test1(model="1"):


    pcd = PointCloudData(        
        raw_rgb=models[model]["rgb_path"],
        raw_depth=models[model]["depth_path"],
        mask=models[model]["mask_path"],
        camera_info=models[model]["scene_info_json"],
        nearest_neighbor=500,
        distance_thresh=0.005,
        semght_threshold=10,
        debug = False
    )

    cloudSegments = pcd.getCloudSegments() #uses open3d plane segmentation 

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Target Object Point Cloud")
    print(f"Test1: Displaying {len(cloudSegments)} segments")
    for segment in cloudSegments:
        #display each cloudSegemnt pointcloud as a different color
        colors = np.random.rand(3)
        segment.paint_uniform_color(colors)
        vis.add_geometry(segment)

    # vis.add_geometry(pcd.getPCD())
    opt = vis.get_render_option()
    opt.line_width = 20

    # Run
    vis.run()
    vis.destroy_window()


def test2(orientation='front'):
    scene_info_json = "../data_video/000001/scene_camera.json"
    with open(scene_info_json, "r") as f:
        scene_info = json.load(f)

    cam_data = scene_info["0"]
    fx, fy = cam_data["cam_K"][0], cam_data["cam_K"][4]
    cx, cy = cam_data["cam_K"][2], cam_data["cam_K"][5]

    # vis = o3d.visualization.Visualizer()
    # vis.create_window(window_name="RGB-D Sequence with Grasp Poses")
    
    # # Create geometries
    # pcd_geometry = o3d.geometry.PointCloud()
    # grasp_geometry = o3d.geometry.TriangleMesh()  # Initialize as empty mesh
    
    # # Add the empty geometries to the visualizer
    # vis.add_geometry(pcd_geometry)
    # vis.add_geometry(grasp_geometry)

    # opt = vis.get_render_option()
    # opt.point_size = 2.0
    # opt.line_width = 5.0  # For coordinate frame visualization

    first_frame = True
    frame_count = 0

    for i in range(1826):
        print(f"Processing frame {i}")
        rgb_path = f"../data_video/000001/rgb/{i:06d}.png"
        depth_path = f"../data_video/000001/depth/{i:06d}.png"
        mask_path = f"../data_video/000001/mask/{i:06d}_000000.png"

        if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
            continue

        try:

            with profile_block("Superquadric Grasp Generation"):
                # Create PointCloudData object
                pcd_data = PointCloudData(        
                    raw_rgb = rgb_path,
                    raw_depth = depth_path,
                    mask = mask_path,
                    camera_info = scene_info,
                    nearest_neighbor=500,
                    distance_thresh=0.005,
                    semght_threshold=10,
                    debug = False
                )
                
                # Get point cloud and segments
                current_pcd = pcd_data.getPCD()
                cloudSegments = pcd_data.getCloudSegments()

                superquadrics = []

                # Generate superquadric fits for each segment
                for segment in cloudSegments:
                    if len(segment.points) > 10:  # Minimum points check
                        sq = Superquadric(segment, downsample=30, debug=False)
                        superquadrics.append(sq)

                # Generate and select grasp from superquadrics
                grasps_obj = Grasps(
                    superquadrics, 
                    target_frame=None,  # Fixed parameter name
                    object_center = pcd_data.getCenter(),
                    orientation= orientation, 
                    grasp_width= 0.236, 
                    debug= False
                )
                
                # Get the grasp pose (coordinate frame)
                new_grasp_pose = grasps_obj.getSelectedGrasps()

                #end tests, display results
            print(f"Frame {i}:")
            # Update grasp geometry
            # if new_grasp_pose is not None:
            #     # Remove old grasp visualization
            #     # vis.remove_geometry(grasp_geometry, reset_bounding_box=False)
                
            #     # Use the new grasp pose
            #     grasp_geometry = new_grasp_pose
                
                # Add new grasp visualization
                # vis.add_geometry(grasp_geometry, reset_bounding_box=False)

            # Update point cloud
            # pcd_geometry.points = current_pcd.points
            # pcd_geometry.colors = current_pcd.colors

            # # Set the view once (only first frame with points)
            # if first_frame and len(current_pcd.points) > 0:
            #     vis.reset_view_point(True)  # Reset to see all geometry
            #     first_frame = False

            # vis.update_geometry(pcd_geometry)
            # vis.poll_events()
            # vis.update_renderer()
            
            # frame_count += 1
            # print(f"Frame {i}: Processed {len(cloudSegments)} segments, {len(current_pcd.points)} points")
            
        except Exception as e:
            print(f"Frame {i}: Error - {e}")
            continue


    # print(f"Processed {frame_count} frames total")
    # vis.destroy_window()


if __name__ == "__main__":
    test1()
