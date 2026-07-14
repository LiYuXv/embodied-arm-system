cmake_minimum_required(VERSION 3.8)
project(embodied_simulation)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY
    launch
    worlds
    config
    urdf
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
