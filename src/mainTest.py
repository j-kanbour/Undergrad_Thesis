
from pointCloudData import PointCloudData
from superquadric import Superquadric
from pointCloudData import PointCloudData
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
def test1(model):

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

    cloudSegments = pcd.getCloudSegments() #uses open3d plane segmentation 
    # print(save_pointcloud_to_ply(pcd.getPCD(), "model_1.ply"))
    # return
    pcd_time = time.time()

    # vis = o3d.visualization.Visualizer()
    # vis.create_window(window_name="Target Object Point Cloud")
    # vis.add_geometry(pcd.getPCD())

    # # #add voxel visual mere
    # # vis.add_geometry(voxelGrid)

    # # # Style
    # # opt = vis.get_render_option()
    # # opt.line_width = 20

    # # vis.run()
    # # vis.destroy_window()
    # # return
    
    # sdf, voxel_grid = utils.generate_sdf_for_mps(
    #     pcd.getPCD(), 
    #     method='distance_based',  # or 'voxel_grid' 
    #     voxel_size=0.01,
    #     truncation_distance=0.01
    # )

    # if sdf.any() and voxel_grid is not None: 
    #     print("SDF and voxel grid generated successfully.")

    #     # ============ VOXEL GRID VISUALIZATION OPTIONS ============
        
    #     # Option 1: Show all voxel centers as a point cloud
    #     voxel_centers_pcd = o3d.geometry.PointCloud()
    #     voxel_centers_pcd.points = o3d.utility.Vector3dVector(voxel_grid['points'].T)
    #     voxel_centers_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Gray color
        
    #     # Option 2: Show only inside voxels (negative SDF)
    #     inside_mask = sdf < 0
    #     inside_points = voxel_grid['points'][:, inside_mask]
    #     inside_pcd = o3d.geometry.PointCloud()
    #     inside_pcd.points = o3d.utility.Vector3dVector(inside_points.T)
    #     inside_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red for inside
        
    #     # Option 3: Show only surface voxels (near zero SDF)
    #     surface_mask = np.abs(sdf) < voxel_grid['truncation'] * 0.5
    #     surface_points = voxel_grid['points'][:, surface_mask]
    #     surface_pcd = o3d.geometry.PointCloud()
    #     surface_pcd.points = o3d.utility.Vector3dVector(surface_points.T)
    #     surface_pcd.paint_uniform_color([0.0, 1.0, 0.0])  # Green for surface
        
    #     # Visualize together
    #     vis = o3d.visualization.Visualizer()
    #     vis.create_window(window_name="Voxel Grid Visualization")
    #     vis.add_geometry(pcd.getPCD())
    #     vis.add_geometry(voxel_centers_pcd)  # Uncomment to see all voxels
    #     vis.add_geometry(inside_pcd)         # Show inside voxels
    #     vis.add_geometry(surface_pcd)        # Show surface voxels
        
    #     vis.run()
    #     vis.destroy_window()

    # file_path = r"obj_000021_normalized.csv"
    # if not file_path:
    #     raise ValueError("No file selected.")

    # # 读取CSV文件
    # sdf = np.genfromtxt(file_path, delimiter=',').T
    # voxelGrid = {}

    # # 设置体素网格参数
    # voxelGrid['size'] = np.ones(3, dtype=int) * int(sdf[0])
    # voxelGrid['range'] = sdf[1:7]
    # sdf = sdf[7:]
    # # 创建线性空间
    # voxelGrid['x'] = np.linspace(voxelGrid['range'][0], voxelGrid['range'][1], int(voxelGrid['size'][0]))
    # voxelGrid['y'] = np.linspace(voxelGrid['range'][2], voxelGrid['range'][3], int(voxelGrid['size'][1]))
    # voxelGrid['z'] = np.linspace(voxelGrid['range'][4], voxelGrid['range'][5], int(voxelGrid['size'][2]))

    # # 创建网格
    # x, y, z = np.meshgrid(voxelGrid['x'], voxelGrid['y'], voxelGrid['z'], indexing='ij')
    # points = np.stack((x,y,z),axis=3)
    # voxelGrid['points'] = points.reshape((-1,3),order='F').T 

    # # 计算间隔和截断
    # voxelGrid['interval'] = (voxelGrid['range'][1] - voxelGrid['range'][0]) / (voxelGrid['size'][0] - 1)
    # voxelGrid['truncation'] = 1.2 * voxelGrid['interval']
    # voxelGrid['disp_range'] = [-np.inf, voxelGrid['truncation']]
    # voxelGrid['visualizeArclength'] = 0.01 * np.sqrt(voxelGrid['range'][1] - voxelGrid['range'][0])

    # # 截断SDF
    # sdf = np.clip(sdf, -voxelGrid['truncation'], voxelGrid['truncation'])
    sdf_voxel_time = time.time()
    
    # Then use with MPS:
    x = mps(sdf, voxelGrid)
    print(f"number of superquadrics: {len(x)}")

    mps_time = time.time()

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Target Object Point Cloud")

    total_mesh = o3d.geometry.TriangleMesh()
    for quadric in x:
        # Build superquadric object
        sq = superquadric(
            quadric[0:2],    # shape
            quadric[2:5],    # scale
            quadric[5:8],    # euler
            quadric[8:11]    # translation
        )

        # Get sq mesh
        mesh = sq.showSuperquadrics()
        
        # Combine meshes
        total_mesh += mesh

    sq_time = time.time()
    
    vis.add_geometry(pcd.getPCD())
    vis.add_geometry(total_mesh)


    # Style
    # vis.add_geometry(pcd.getPCD())
    opt = vis.get_render_option()
    opt.line_width = 20

    print(f"\n\n\n\n\
        start time: {init_time} \n\
        pcd_time: {pcd_time - init_time} \n\
        sdf_voxel_time: {sdf_voxel_time - pcd_time} \n\
        mps_time: {mps_time - sdf_voxel_time} \n\
        sq_time: {sq_time - mps_time} \n\
        total time: {sq_time - init_time} \n\
        \n\n\n\n\n")

    # Run
    vis.run()
    vis.destroy_window()

""" Test 2: graps """
def test2(model):

    # CPU profiling start
    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None)  # prime

    cpu_start = process.cpu_times()
    num_threads_before = process.num_threads()
    start_wall = time.perf_counter()

    # Core operation
    superquadric = Superquadric(
        object_ID=models[model]["object_ID"],
        class_name=models[model]["class_name"],
        raw_rgb=models[model]["rgb_path"],
        raw_depth=models[model]["depth_path"],
        mask=models[model]["mask_path"],
        camera_info=models[model]["scene_info_json"]
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
