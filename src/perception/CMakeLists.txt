cmake_minimum_required(VERSION 2.8.3)
project(icra20_manipulation_pose)

set(CMAKE_BUILD_TYPE Release)
add_compile_options(-std=c++11 -fopenmp)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  shape_msgs
  geometry_msgs
  sensor_msgs
  std_msgs
  message_generation
  cv_bridge
  pcl_ros
)
find_package(Boost REQUIRED COMPONENTS system program_options)
find_package(PCL 1.9 REQUIRED  COMPONENTS io common visualization features registration)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(BLAS REQUIRED )

## Generate added messages and services with any dependencies listed here
generate_messages(
 DEPENDENCIES
 geometry_msgs
 sensor_msgs
 std_msgs
)


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${CATKIN_DEVEL_PREFIX}/include
  ${OpenCV_INCLUDE_DIRS}
  ${BLAS_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/../depth_sim/include
  ${PROJECT_SOURCE_DIR}/../OpenGR_4pcs/src
  ${PROJECT_SOURCE_DIR}/../OpenGR_4pcs/demos/PCLWrapper
)
file(GLOB_RECURSE OPENGR_SRC  ${PROJECT_SOURCE_DIR}/../OpenGR_4pcs/src/*.hpp ${PROJECT_SOURCE_DIR}/../OpenGR_4pcs/demos/PCLWrapper/pcl/registration/impl/super4pcs.hpp)
file(GLOB MY_SRC ${PROJECT_SOURCE_DIR}/src/*.cpp)
add_library(${PROJECT_NAME} ${MY_SRC} ${OPENGR_SRC})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES} ${OpenMP_CXX_FLAGS}
${PROJECT_SOURCE_DIR}/../../devel/lib/libbowen_depth_sim.so ${YAML_CPP_LIBRARIES}  ${BLAS_LIBRARIES})
catkin_package(
  INCLUDE_DIRS
    include
  DEPENDS
    PCL
  CATKIN_DEPENDS
    std_msgs
    geometry_msgs

  LIBRARIES
  ${PROJECT_NAME}
)
install(TARGETS ${PROJECT_NAME}
LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})
add_executable(computePPF  src/app/computePPF.cpp)
target_link_libraries(computePPF ${PROJECT_NAME})
add_executable(main_realdata_auto  src/app/main_realdata_auto.cpp)
target_link_libraries(main_realdata_auto ${PROJECT_NAME})
add_executable(run_real_all  src/app/run_real_all.cpp)
target_link_libraries(run_real_all ${PROJECT_NAME})

