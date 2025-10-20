cmake_minimum_required(VERSION 3.0.2)
project(generate_grasp)

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
  message_generation  # Required for service generation
)

find_package(OpenCV REQUIRED)

# Add service files
add_service_files(
  FILES
  grasp_msg.srv
)

# catkin_python_setup() MUST come BEFORE generate_messages()
catkin_python_setup()

# Generate messages AFTER catkin_python_setup()
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

# catkin_package() comes after generate_messages()
catkin_package(
  CATKIN_DEPENDS
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
    message_runtime  # Required at runtime for generated services
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# Install executable Python nodes
catkin_install_python(
  PROGRAMS
    scripts/grasp_service.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install launch files
install(
  DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)