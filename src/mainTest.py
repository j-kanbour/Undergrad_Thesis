
from pointCloudData import PointCloudData
from superquadric import Superquadric
import open3d as o3d
import sys, os, time
import psutil
# from grasps import Grasps
import numpy as np

""" Test 1: modelling"""
def test1(model):

    #load data models
    if model == "1":
        rgb_path = "../data/rgb_and_depth_data/000001/rgb/000000.png"
        depth_path = "../data/rgb_and_depth_data/000001/depth/000000.png"
        mask_path = "../data/rgb_and_depth_data/000001/mask_visib/000000_000000.png"
        scene_info_json = ["../data/rgb_and_depth_data/000001/scene_camera.json","0"]
        class_name="bottle"
        object_ID = 1

    elif model == "2":
        rgb_path = "../data/rgb_and_depth_data/000001/rgb/000001.png"
        depth_path = "../data/rgb_and_depth_data/000001/depth/000001.png"
        mask_path = "../data/rgb_and_depth_data/000001/mask_visib/000000_000001.png"
        scene_info_json = ["../data/rgb_and_depth_data/000001/scene_camera.json","1"]
        class_name="can"
        object_ID = 2

    elif model == "3":
        rgb_path = "../data/rgb_and_depth_data/000008/rgb/000000.png"
        depth_path = "../data/rgb_and_depth_data/000008/depth/000000.png"
        mask_path = "../data/rgb_and_depth_data/000008/mask_visib/000001_000001.png"
        scene_info_json = ["../data/rgb_and_depth_data/000008/scene_camera.json","0"]
        class_name="can"
        object_ID = 3

    elif model == "4":
        rgb_path = "../data/rgb_and_depth_data/000005/rgb/000000.png"
        depth_path = "../data/rgb_and_depth_data/000005/depth/000000.png"
        mask_path = "../data/rgb_and_depth_data/000005/mask_visib/000001_000001.png"
        scene_info_json = ["../data/rgb_and_depth_data/000005/scene_camera.json","0"]
        class_name="box"
        object_ID = 3

    elif model == "5":
        rgb_path = "../data/rgb_and_depth_data/000005/rgb/000004.png"
        depth_path = "../data/rgb_and_depth_data/000005/depth/000004.png"
        mask_path = "../data/rgb_and_depth_data/000005/mask_visib/000004_000010.png"
        scene_info_json = ["../data/rgb_and_depth_data/000005/scene_camera.json","4"]
        object_ID = 3
    
    # Core operation
    superquadric = Superquadric(
        object_ID=object_ID,
        class_name=class_name,
        raw_rgb=rgb_path,
        raw_depth=depth_path,
        mask=mask_path,
        camera_info=scene_info_json
    )

    # Visualisation
    pointcloud = superquadric.pcd

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Target Object Point Cloud")

    vis.add_geometry(pointcloud.getPCD())
    vis.add_geometry(superquadric.getSuperquadricAsPCD())
    # vis.add_geometry(superquadric.getAlignedPCD())

    opt = vis.get_render_option()
    opt.line_width = 20

    vis.run()
    vis.destroy_window()

""" Test 2: graps """
def test2(model):
    
    #load data models
    if model == "1":
        rgb_path = "../data/rgb_and_depth_data/000001/rgb/000000.png"
        depth_path = "../data/rgb_and_depth_data/000001/depth/000000.png"
        mask_path = "../data/rgb_and_depth_data/000001/mask_visib/000000_000000.png"
        scene_info_json = ["../data/rgb_and_depth_data/000001/scene_camera.json","0"]
        class_name="bottle"
        object_ID = 1

    elif model == "2":
        rgb_path = "../data/rgb_and_depth_data/000001/rgb/000001.png"
        depth_path = "../data/rgb_and_depth_data/000001/depth/000001.png"
        mask_path = "../data/rgb_and_depth_data/000001/mask_visib/000000_000001.png"
        scene_info_json = ["../data/rgb_and_depth_data/000001/scene_camera.json","1"]
        class_name="can"
        object_ID = 2

    elif model == "3":
        rgb_path = "../data/rgb_and_depth_data/000008/rgb/000000.png"
        depth_path = "../data/rgb_and_depth_data/000008/depth/000000.png"
        mask_path = "../data/rgb_and_depth_data/000008/mask_visib/000001_000001.png"
        scene_info_json = ["../data/rgb_and_depth_data/000008/scene_camera.json","0"]
        class_name="can"
        object_ID = 3

    elif model == "4":
        rgb_path = "../data/rgb_and_depth_data/000005/rgb/000000.png"
        depth_path = "../data/rgb_and_depth_data/000005/depth/000000.png"
        mask_path = "../data/rgb_and_depth_data/000005/mask_visib/000001_000001.png"
        scene_info_json = ["../data/rgb_and_depth_data/000005/scene_camera.json","0"]
        class_name="box"
        object_ID = 3

    elif model == "5":
        rgb_path = "../data/rgb_and_depth_data/000005/rgb/000004.png"
        depth_path = "../data/rgb_and_depth_data/000005/depth/000004.png"
        mask_path = "../data/rgb_and_depth_data/000005/mask_visib/000004_000010.png"
        scene_info_json = ["../data/rgb_and_depth_data/000005/scene_camera.json","4"]
        object_ID = 3
    # CPU profiling start
    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None)  # prime

    cpu_start = process.cpu_times()
    num_threads_before = process.num_threads()
    start_wall = time.perf_counter()

    # Core operation
    superquadric = Superquadric(
        object_ID=object_ID,
        class_name=class_name,
        raw_rgb=[rgb_path],
        raw_depth=[depth_path],
        mask=[mask_path],
        camera_info=scene_info_json
    )

    end_wall = time.perf_counter()
    cpu_end = process.cpu_times()
    num_threads_after = process.num_threads()

    # Deltas
    user_cpu = cpu_end.user - cpu_start.user
    system_cpu = cpu_end.system - cpu_start.system
    wall_time = end_wall - start_wall
    total_cpu = user_cpu + system_cpu
    cpu_percent = (total_cpu / wall_time) * 100 if wall_time > 0 else 0

    print("\n===== CPU Usage for Superquadric Creation =====")
    print(f"Wall time elapsed: {wall_time:.4f} seconds")
    print(f"User CPU time:     {user_cpu:.4f} seconds")
    print(f"System CPU time:   {system_cpu:.4f} seconds")
    print(f"Total CPU usage:   {cpu_percent:.1f}% of one core")
    print(f"Threads before:    {num_threads_before}, after: {num_threads_after}")
    print("==============================================\n")

    # Visualisation setup
    pointcloud = superquadric.pcd

    centroid_coords = pointcloud.getCentroid()
    centroid = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    centroid.paint_uniform_color([1, 0, 0])
    centroid.translate(centroid_coords)

    bbox = o3d.geometry.LineSet.create_from_oriented_bounding_box(pointcloud.getBoundingBox())
    bbox.paint_uniform_color([0, 1, 0])

    # Align superquadric
    aligned_superquadric_pcd = superquadric.getAlignedPCD()

    # ===== GRASP GENERATION =====
    grasps_generator = Grasps(superquadric)


    grasps = grasps_generator.getSelectedGrasps()
    print(grasps)

    # Final visualisation
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Target Object Point Cloud + Grasps")

    offset = np.array([0, 0, -0.05])  # 5 cm behind object along camera Z
    frame_position = centroid_coords + offset
    object_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    object_frame.translate(frame_position)
    vis.add_geometry(object_frame)

    bbox = o3d.geometry.LineSet.create_from_oriented_bounding_box(pointcloud.getBoundingBox())
    bbox.paint_uniform_color([0, 1, 0])

    # Add object and aligned superquadric
    vis.add_geometry(pointcloud.getPCD())
    vis.add_geometry(aligned_superquadric_pcd)

    # Add axis
    vis.add_geometry(centroid)

    # Add bounding box
    vis.add_geometry(bbox)

    # Prepare and add grasp point spheres + axes (draw together to match)
    axis_scale = 0.1  # length of axis lines

    if grasps == None: 
        print("no possible grassps")
    else:
        for grasp in grasps:
            # ===== Show normal at point_i =====
            point_i = grasp["point_i"]
            n_i = grasp["point_i_normals"]
            normal_length = axis_scale * 0.5  # shorter for normals

            n_i_end = point_i + n_i * normal_length
            n_i_arrow = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector([point_i, n_i_end]),
                lines=o3d.utility.Vector2iVector([[0, 1]])
            )
            n_i_arrow.paint_uniform_color([1, 0, 0])  # yellow normal
            vis.add_geometry(n_i_arrow)

            # ===== Show normal at point_j =====
            point_j = grasp["point_j"]
            n_j = grasp["point_j_normals"]

            n_j_end = point_j + n_j * normal_length
            n_j_arrow = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector([point_j, n_j_end]),
                lines=o3d.utility.Vector2iVector([[0, 1]])
            )
            n_j_arrow.paint_uniform_color([1, 0, 1])  # yellow normal
            vis.add_geometry(n_j_arrow)

            # ===== Optionally, still show point_i and point_j =====
            point_i_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
            point_i_sphere.paint_uniform_color([1, 0, 1])  # magenta
            point_i_sphere.translate(point_i)
            vis.add_geometry(point_i_sphere)

            point_j_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
            point_j_sphere.paint_uniform_color([1, 0, 1])  # magenta
            point_j_sphere.translate(point_j)
            vis.add_geometry(point_j_sphere)


    # Visual options
    opt = vis.get_render_option()
    opt.line_width = 20

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    test_functions = {
        "1": test1,
        "2": test2
    }
    if len(sys.argv) != 3:
        print("add test number as argument: python3 mainTest.py 2")
    else:
        test_number = str(sys.argv[1])
        model = sys.argv[2]
        if test_number in test_functions:
            test_functions[test_number](model)
        else:
            print(f"No test function defined for test{test_number}")
    
