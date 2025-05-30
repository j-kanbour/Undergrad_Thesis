from pointCloudData import PointCloudData
from superquadric import Superquadric
import open3d as o3d
import sys

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
def test5():
    rgb_path="data/rgb_and_depth_data/000001/rgb/000000.png",
    depth_path="data/rgb_and_depth_data/000001/depth/000000.png",
    mask_path="data/rgb_and_depth_data/000001/mask_visib/000000_000000.png",
    scene_info_json="data/rgb_and_depth_data/000001/scene_camera.json"

    superquadric = Superquadric(object_ID=1, class_name="Bottle", input_type="RGB and DEPTH", raw_data_1=rgb_path, raw_depth=depth_path, raw_mask=mask_path, camera_info=scene_info_json)
    pointcloud = superquadric.getPCD()
    
    # Create a red sphere at the centroid
    centroid_coords = pointcloud.getCentroid()
    centroid = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    centroid.paint_uniform_color([1, 0, 0])
    centroid.translate(centroid_coords)

    bbox = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(pointcloud.getBoundingBox())
    bbox.paint_uniform_color([0, 1, 0])

    o3d.visualization.draw_geometries(
        [pointcloud.getPCD(), centroid, bbox],
        window_name="Target Object Point Cloud"
    )


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
        "5": test5
    }
    if len(sys.argv) != 2:
        print("add test number as argument: python3 mainTest.py 2")
    else:
        test_number = str(sys.argv[1])
        if test_number in test_functions:
            test_functions[test_number]()
        else:
            print(f"No test function defined for test{test_number}")
    
