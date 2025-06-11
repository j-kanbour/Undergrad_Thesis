from pointCloudData import PointCloudData
from superquadric import Superquadric
import open3d as o3d
import sys, os, time
import psutil
from grasps import Grasps
import numpy as np

""" metrics

Superquadric Fitting:


6D Pose Estimation:


Grasp Selection:
"""

""" Test 1

Input:  
- given a point cloid image of an object
- the object is in isolation
- the point cloud is a complete model of the obejct

Output:
- A single superquadric that provides the best possible model for the given object
"""


""" Test 2

Input:  
- given a point cloid image of an object
- the object is in isolation
- the point cloud is a partial model of the obejct from a single perspective

Output:
- A single superquadric that provides the best possible model for the given object
"""


""" Test 3

Input:  
- given separate RGB and Depth image of an object
- the object is in isolation
- the images are from a single perspective

Output:
- A single superquadric that provides the best possible model for the given object
"""


""" Test 4

Input:  
- given a single RGB-D image of an object
- the object is in isolation
- the image is from a single perspective

Output:
- A single superquadric that provides the best possible model for the given object
"""


""" Test 5

Input:  
- given separate RGB and Depth image of a scene
- there will be only one target object
- the target object is in not in isolation
- the images are from a single perspective

Additional Processes: 
- YOLOv11 will be used to isolate the object 
- The process will provide a bitwise mask of the object in scene

Output:
- A single superquadric that provides the best possible model for the given object
"""
def test5(model):
    

    rgb_path = "data/rgb_and_depth_data/000001/rgb/000000.png"
    depth_path = "data/rgb_and_depth_data/000001/depth/000000.png"
    mask_path = "data/rgb_and_depth_data/000001/mask_visib/000000_000000.png"
    scene_info_json = "data/rgb_and_depth_data/000001/scene_camera.json"
    class_name="Bottle"
    object_ID = 1

    if model == "2":
        rgb_path = "data/rgb_and_depth_data/000001/rgb/000001.png"
        depth_path = "data/rgb_and_depth_data/000001/depth/000001.png"
        mask_path = "data/rgb_and_depth_data/000001/mask_visib/000000_000001.png"
        scene_info_json = "data/rgb_and_depth_data/000001/scene_camera.json"
        class_name="Can"
        object_ID = 2

    if model == "3":
        rgb_path = "data/rgb_and_depth_data/000008/rgb/000000.png"
        depth_path = "data/rgb_and_depth_data/000008/depth/000000.png"
        mask_path = "data/rgb_and_depth_data/000008/mask_visib/000001_000001.png"
        scene_info_json = "data/rgb_and_depth_data/000008/scene_camera.json"
        class_name="Can"
        object_ID = 3

    if model == "4":
        rgb_path = "data/rgb_and_depth_data/000005/rgb/000000.png"
        depth_path = "data/rgb_and_depth_data/000005/depth/000000.png"
        mask_path = "data/rgb_and_depth_data/000005/mask_visib/000001_000001.png"
        scene_info_json = "data/rgb_and_depth_data/000005/scene_camera.json"
        object_ID = 3

    if model == "5":
        rgb_path = "data/rgb_and_depth_data/000005/rgb/000004.png"
        depth_path = "data/rgb_and_depth_data/000005/depth/000004.png"
        mask_path = "data/rgb_and_depth_data/000005/mask_visib/000004_000010.png"
        scene_info_json = "data/rgb_and_depth_data/000005/scene_camera.json"
        object_ID = 3
    
        
    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None)  # prime

    cpu_start = process.cpu_times()
    num_threads_before = process.num_threads()
    start_wall = time.perf_counter()

    # Core operation
    superquadric = Superquadric(
        object_ID=object_ID,
        class_name=class_name,
        input_type="RGB and DEPTH",
        raw_data_1=[rgb_path],
        raw_depth=[depth_path],
        raw_mask=[mask_path],
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

    # Visualisation
    pointcloud = superquadric.pcd

    centroid_coords = pointcloud.getCentroid()
    centroid = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    centroid.paint_uniform_color([1, 0, 0])
    centroid.translate(centroid_coords)

    bbox = o3d.geometry.LineSet.create_from_oriented_bounding_box(pointcloud.getBoundingBox())
    bbox.paint_uniform_color([0, 1, 0])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Target Object Point Cloud")

    vis.add_geometry(pointcloud.getPCD())
    #vis.add_geometry(superquadric.getSuperquadricAsPCD())
    vis.add_geometry(superquadric.alignWithICP())
    vis.add_geometry(centroid)
    vis.add_geometry(bbox)

    opt = vis.get_render_option()
    opt.line_width = 20

    vis.run()
    vis.destroy_window()


""" Test 6

Input:  
- given a single RGB-D image of a scene
- there will be only one target object
- the target object is in not in isolation
- the image is from a single perspective

Additional Processes: 
- YOLOv11 will be used to isolate the object 
- The process will provide a bitwise mask of the object in scene

Output:
- A single superquadric that provides the best possible model for the given object
"""
def test6(model):

    rgb_path = "data/rgb_and_depth_data/000001/rgb/000000.png"
    depth_path = "data/rgb_and_depth_data/000001/depth/000000.png"
    mask_path = "data/rgb_and_depth_data/000001/mask_visib/000000_000000.png"
    scene_info_json = "data/rgb_and_depth_data/000001/scene_camera.json"
    object_ID = 1

    if model == "2":
        rgb_path = "data/rgb_and_depth_data/000001/rgb/000001.png"
        depth_path = "data/rgb_and_depth_data/000001/depth/000001.png"
        mask_path = "data/rgb_and_depth_data/000001/mask_visib/000000_000001.png"
        scene_info_json = "data/rgb_and_depth_data/000001/scene_camera.json"
        object_ID = 2

    if model == "3":
        rgb_path = "data/rgb_and_depth_data/000008/rgb/000000.png"
        depth_path = "data/rgb_and_depth_data/000008/depth/000000.png"
        mask_path = "data/rgb_and_depth_data/000008/mask_visib/000001_000001.png"
        scene_info_json = "data/rgb_and_depth_data/000008/scene_camera.json"
        object_ID = 3

    if model == "4":
        rgb_path = "data/rgb_and_depth_data/000005/rgb/000000.png"
        depth_path = "data/rgb_and_depth_data/000005/depth/000000.png"
        mask_path = "data/rgb_and_depth_data/000005/mask_visib/000001_000001.png"
        scene_info_json = "data/rgb_and_depth_data/000005/scene_camera.json"
        object_ID = 3

    if model == "5":
        rgb_path = "data/rgb_and_depth_data/000005/rgb/000004.png"
        depth_path = "data/rgb_and_depth_data/000005/depth/000004.png"
        mask_path = "data/rgb_and_depth_data/000005/mask_visib/000004_000010.png"
        scene_info_json = "data/rgb_and_depth_data/000005/scene_camera.json"
        object_ID = 3

    # CPU profiling start
    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None)
    cpu_start = process.cpu_times()
    num_threads_before = process.num_threads()
    start_wall = time.perf_counter()

    # Core operation — create superquadric
    superquadric = Superquadric(
        object_ID=object_ID,
        class_name="Bottle",
        input_type="RGB and DEPTH",
        raw_data_1=[rgb_path],
        raw_depth=[depth_path],
        raw_mask=[mask_path],
        camera_info=scene_info_json
    )

    end_wall = time.perf_counter()
    cpu_end = process.cpu_times()
    num_threads_after = process.num_threads()

    # CPU profiling report
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

    bbox = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(pointcloud.getBoundingBox())
    bbox.paint_uniform_color([0, 1, 0])

    axis_data = pointcloud.findAxis()
    if axis_data:
        origin = axis_data["origin"]
        direction = axis_data["direction"]
        length = 0.05
        start = origin - direction * length
        end = origin + direction * length
        axis_line = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([start, end]),
            lines=o3d.utility.Vector2iVector([[0, 1]])
        )
        axis_line.paint_uniform_color([0, 0, 1])
    else:
        axis_line = o3d.geometry.LineSet()

    # Align superquadric
    aligned_superquadric_pcd = superquadric.alignWithICP()

    # ===== GRASP GENERATION =====
    grasps_generator = Grasps(aligned_superquadric_pcd)
    grasps = grasps_generator.generate_antipodal_grasps(num_grasps=5)

    # Print grasps as proper two-finger gripper pose
    for i, grasp in enumerate(grasps):
        print(f"\n=== Grasp {i+1} ===")
        print("Position:", grasp["position"])
        print("Approach axis (z):", grasp["approach_axis"])
        print("Binormal axis (y):", grasp["binormal_axis"])
        print("Axis between fingers (x):", grasp["axis"])
        print("Jaw width:", grasp["jaw_width"])
        print("Orientation matrix:\n", grasp["orientation"])

    # Prepare visualisation of grasp axes
    grasp_axes = []
    for grasp in grasps:
        position = grasp["position"]
        orientation = grasp["orientation"]
        scale = 0.1   # length of axis lines

        # X axis (axis between fingers) → red
        x_end = position + orientation[:, 0] * scale
        x_axis_line = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([position, x_end]),
            lines=o3d.utility.Vector2iVector([[0, 1]])
        )
        x_axis_line.paint_uniform_color([1, 0, 0])
        grasp_axes.append(x_axis_line)

        # Y axis (binormal axis) → green
        y_end = position + orientation[:, 1] * scale
        y_axis_line = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([position, y_end]),
            lines=o3d.utility.Vector2iVector([[0, 1]])
        )
        y_axis_line.paint_uniform_color([0, 1, 0])
        grasp_axes.append(y_axis_line)

        # Z axis (approach axis) → blue
        z_end = position + orientation[:, 2] * scale
        z_axis_line = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([position, z_end]),
            lines=o3d.utility.Vector2iVector([[0, 1]])
        )
        z_axis_line.paint_uniform_color([0, 0, 1])
        grasp_axes.append(z_axis_line)

    # Final visualisation
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Target Object Point Cloud + Grasps")

    vis.add_geometry(pointcloud.getPCD())
    vis.add_geometry(aligned_superquadric_pcd)
    vis.add_geometry(centroid)
    vis.add_geometry(bbox)
    vis.add_geometry(axis_line)

    # Add grasp axes
    for axis in grasp_axes:
        vis.add_geometry(axis)

    opt = vis.get_render_option()
    opt.line_width = 50

    vis.run()
    vis.destroy_window()


""" Test 7

Input:  
- given separate RGB and Depth image of a scene
- there will be multiple target objects of the same class
- the target objects are not in isolation
- the images are from a single perspective

Additional Processes: 
- YOLOv11 will be used to isolate the objects
- The process will provide a bitwise mask of the object in scene

Output:
- A single superquadric that provides the best possible model for the given object
"""


""" Test 8

Input:  
- given a single RGB-D image of a scene
- there will be multiple target objects of the same class
- the target objects are not in isolation
- the image is from a single perspective

Additional Processes: 
- YOLOv11 will be used to isolate the objects
- The process will provide a bitwise mask of the object in scene

Output:
- A single superquadric that provides the best possible model for the given object
"""

""" Test 9

Input:  
- given separate RGB and Depth image of a scene
- there will be multiple target objects of multiple classes
- the target objects are not in isolation
- the images are from a single perspective

Additional Processes: 
- YOLOv11 will be used to isolate the objects
- The process will provide a bitwise mask of the object in scene

Output:
- A single superquadric that provides the best possible model for the given object
"""


""" Test 10

Input:  
- given a single RGB-D image of a scene
- there will be multiple target objects of multiple classes
- the target objects are not in isolation
- the image is from a single perspective

Additional Processes: 
- YOLOv11 will be used to isolate the objects
- The process will provide a bitwise mask of the object in scene

Output:
- A single superquadric that provides the best possible model for the given object
"""

""" Test 11

Input:  
- given separate RGB and Depth videos of a scene
- there will be only one target object
- the target object is not in isolation
- the video is from a single perspective

Additional Processes: 
- YOLOv11 will be used to isolate the objects
- The process will provide a bitwise mask of the object in scene

Output:
- A single superquadric that provides the best possible model for the given object
"""

""" Test 12

Input:  
- given a single RGB-D image of a scene
- there will be only one target object
- the target object is not in isolation
- the video is from a single perspective

Additional Processes: 
- YOLOv11 will be used to isolate the objects
- The process will provide a bitwise mask of the object in scene

Output:
- A single superquadric that provides the best possible model for the given object
"""

if __name__ == "__main__":
    test_functions = {
        "5": test5,
        "6": test6
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
    
