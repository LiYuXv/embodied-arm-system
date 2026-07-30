cmake_minimum_required(VERSION 3.8)
project(gazebo_classic_grasp_attachment)

find_package(ament_cmake REQUIRED)
find_package(gazebo REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_srvs REQUIRED)

add_library(gazebo_classic_grasp_attachment SHARED src/grasp_attachment_plugin.cpp)
target_compile_features(gazebo_classic_grasp_attachment PUBLIC cxx_std_17)
target_include_directories(gazebo_classic_grasp_attachment SYSTEM PRIVATE ${GAZEBO_INCLUDE_DIRS})
target_link_libraries(gazebo_classic_grasp_attachment ${GAZEBO_LIBRARIES})
ament_target_dependencies(gazebo_classic_grasp_attachment gazebo_ros rclcpp std_srvs)

install(TARGETS gazebo_classic_grasp_attachment LIBRARY DESTINATION lib)

ament_package()
