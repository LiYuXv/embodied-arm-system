cmake_minimum_required(VERSION 3.8)
project(gazebo_classic_gripper_mimic)

find_package(ament_cmake REQUIRED)
find_package(gazebo REQUIRED)

add_library(gazebo_classic_gripper_mimic SHARED src/gripper_mimic_plugin.cpp)
target_compile_features(gazebo_classic_gripper_mimic PUBLIC cxx_std_17)
target_include_directories(gazebo_classic_gripper_mimic SYSTEM PRIVATE ${GAZEBO_INCLUDE_DIRS})
target_link_libraries(gazebo_classic_gripper_mimic ${GAZEBO_LIBRARIES})

install(TARGETS gazebo_classic_gripper_mimic LIBRARY DESTINATION lib)

ament_package()
