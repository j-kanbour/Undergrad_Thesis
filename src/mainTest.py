
from pointCloudData import PointCloudData
from superquadric import Superquadric
from grasps import Grasps
import open3d as o3d
import sys, os, time
import psutil
import numpy as np
import copy


models = { 
            "1": {
                "rgb_path":"../data/rgb_and_depth_data/000001/rgb/000000.png",
                "depth_path":"../data/rgb_and_depth_data/000001/depth/000000.png",
                "mask_path":"../data/rgb_and_depth_data/000001/mask_visib/000000_000000.png",
                "scene_info_json":["../data/rgb_and_depth_data/000001/scene_camera.json","0"],
                "class_name":"bottle",
                "object_ID":1
            },
            "2": {                
                "rgb_path":"../data/rgb_and_depth_data/000001/rgb/000001.png",
                "depth_path":"../data/rgb_and_depth_data/000001/depth/000002.png",
                "mask_path":"../data/rgb_and_depth_data/000001/mask_visib/000000_000001.png",
                "scene_info_json":["../data/rgb_and_depth_data/000001/scene_camera.json","1"],
                "class_name":"can",
                "object_ID":2
            },
            "3": {
                "rgb_path":"../data/rgb_and_depth_data/000008/rgb/000000.png",
                "depth_path":"../data/rgb_and_depth_data/000008/depth/000000.png",
                "mask_path":"../data/rgb_and_depth_data/000008/mask_visib/000001_000001.png",
                "scene_info_json":["../data/rgb_and_depth_data/000008/scene_camera.json","0"],
                "class_name":"can",
                "object_ID":3
            },
            "4": {                
                "rgb_path":"../data/rgb_and_depth_data/000005/rgb/000000.png",
                "depth_path":"../data/rgb_and_depth_data/000005/depth/000000.png",
                "mask_path":"../data/rgb_and_depth_data/000005/mask_visib/000001_000001.png",
                "scene_info_json":["../data/rgb_and_depth_data/000005/scene_camera.json","0"],
                "class_name":"box",
                "object_ID":4
            },
            "5": {                
                "rgb_path":"../data/rgb_and_depth_data/000005/rgb/000004.png",
                "depth_path":"../data/rgb_and_depth_data/000005/depth/000004.png",
                "mask_path":"../data/rgb_and_depth_data/000005/mask_visib/000004_000010.png",
                "scene_info_json":["../data/rgb_and_depth_data/000005/scene_camera.json","4"],
                "class_name":"bottle",
                "object_ID":5
            }
        }

""" Test 1: modelling"""
def test1(model, orientation=None):

    # Generate Point Cloud Model
    init_time = time.time()
    # generate point cloud model 
    pcd = PointCloudData(
        object_ID=models[model]["object_ID"],
        raw_rgb=models[model]["rgb_path"],
        raw_depth=models[model]["depth_path"],
        mask=models[model]["mask_path"],
        camera_info=models[model]["scene_info_json"]
        )
    
    pcd_time = time.time() 

    cloudSegments = pcd.getCloudSegments() #uses open3d plane segmentation 

    print(f"number of segments: {len(cloudSegments)}")
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(window_name="Target Object Point Cloud")
    # all_cloudSegments = o3d.geometry.PointCloud()
    # for i in cloudSegments:
    #     all_cloudSegments += i

    # vis.add_geometry(all_cloudSegments)    
    # vis.run()
    # vis.destroy_window()
    # return

    cloudSegment_time = time.time()

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Target Object Point Cloud")

    total_mesh = o3d.geometry.PointCloud()
    sq_list = []

    for segment in cloudSegments:
        # Build superquadric object
        sq = Superquadric(segment).getSuperquadricMesh()
        sq_list.append(sq)
        total_mesh += sq

    sq_time = time.time()

    #grasp selection

    grasp = Grasps(sq_list, 'harb_rgb_camera_frame', orientation=orientation, gripper_depth=0.0666, gripper_width=0.236)

    grasp_points = grasp.getGraspPoints()
    grasp_pose = grasp.getSelectedGrasps()
    
    #generate grasp posestamp

    grasp_time = time.time()
    
    vis.add_geometry(pcd.getPCD())
    #vis.add_geometry(total_mesh)
    #vis.add_geometry(grasp_points)
    vis.add_geometry(grasp_pose)
    opt = vis.get_render_option()
    opt.line_width = 20

    print(f"\n\n\n\n\
        start time: {init_time} \n\
        pcd_time: {pcd_time - init_time} \n\
        segmentation_time: {cloudSegment_time - pcd_time} \n\
        sq_time: {sq_time - cloudSegment_time} \n\
        grasp_time: {grasp_time - sq_time} \n\
        total time: {grasp_time - init_time} \n\
        \n\n\n\n\n")

    # Run
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    test_functions = {
        "1": test1
    }
    if len(sys.argv) != 4:
        print("python3 mainTest.py test_number model_number orientation")
    else:
        test_number = str(sys.argv[1])
        model = sys.argv[2]
        orientation = sys.argv[3]
        if test_number in test_functions:
            test_functions[test_number](model, orientation)
        else:
            print(f"No test function defined for test{test_number}")
