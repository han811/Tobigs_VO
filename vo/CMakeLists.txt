cmake_minimum_required(VERSION 3.0.2)
project(vo)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  geometry_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf
  tf2
)

catkin_python_setup()

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES vo
 CATKIN_DEPENDS actionlib geometry_msgs roscpp rospy sensor_msgs std_msgs tf tf2
 DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/save_data/save_data.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

