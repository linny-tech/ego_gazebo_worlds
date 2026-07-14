cmake_minimum_required(VERSION 3.0.2)
project(ego_gazebo_worlds)

find_package(catkin REQUIRED)

catkin_package()

catkin_install_python(
  PROGRAMS
    scripts/generate_ego_world.py
    scripts/generate_forest_world.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY launch worlds config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
