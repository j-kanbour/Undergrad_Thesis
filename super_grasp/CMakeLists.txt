cmake_minimum_required(VERSION 3.0.2)
project(super_grasp)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  visualization_msgs
  tf
  tf2_ros
  tf2_geometry_msgs
  cv_bridge
  unsw_vision_msgs
)

find_package(OpenCV REQUIRED)

# Use setup.py to register/install your Python package (src/super_grasp)
catkin_python_setup()

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# Install executable Python nodes (chmod +x these files)
catkin_install_python(
  PROGRAMS
    scripts/grasp_generator.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Do NOT also install src/super_grasp here; setup.py handles that.

# Install launch files
install(
  DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)
