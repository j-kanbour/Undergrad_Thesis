cmake_minimum_required(VERSION 3.0.2)
project(super_grasp)

# C++14 required for Open3D
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  geometry_msgs
  visualization_msgs
  tf
  tf2_ros
  tf2_geometry_msgs
  cv_bridge
  unsw_vision_msgs
  image_transport
)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Open3D REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}_lib
  CATKIN_DEPENDS 
    roscpp
    std_msgs
    sensor_msgs
    geometry_msgs
    visualization_msgs
    tf
    tf2_ros
    tf2_geometry_msgs
    cv_bridge
    unsw_vision_msgs
  DEPENDS 
    OpenCV
    EIGEN3
    Open3D
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${Open3D_INCLUDE_DIRS}
)

# Library containing the main classes
add_library(${PROJECT_NAME}_lib
  src/point_cloud_data.cpp
  src/superquadric.cpp
  src/grasps.cpp
)

target_link_libraries(${PROJECT_NAME}_lib
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Open3D_LIBRARIES}
)

# Main executable
add_executable(grasp_generator_node
  src/grasp_generator_node.cpp
)

target_link_libraries(grasp_generator_node
  ${PROJECT_NAME}_lib
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Open3D_LIBRARIES}
)

# Install targets
install(TARGETS ${PROJECT_NAME}_lib grasp_generator_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)