cmake_minimum_required(VERSION 2.8.3)
project(bowen_depth_sim)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)
set(CMAKE_BUILD_TYPE Release)
add_compile_options(-std=c++11 -fopenmp)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Boost REQUIRED)
find_package(catkin REQUIRED COMPONENTS roscpp roslib pcl_ros cv_bridge)
find_package(VTK)
find_package(PCL 1.9 REQUIRED COMPONENTS io common visualization features)

include_directories(${PROJECT_SOURCE_DIR}/include ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS}
  ${VTK_INCLUDE_DIRS})

IF (WIN32)
FIND_PATH( GLEW_INCLUDE_PATH GL/glew.h
           $ENV{PROGRAMFILES}/GLEW/include
           ${PROJECT_SOURCE_DIR}/src/nvgl/glew/include
           DOC "The directory where GL/glew.h resides")
FIND_LIBRARY( GLEW_LIBRARY
              NAMES glew GLEW glew32 glew32s
              PATHS
              $ENV{PROGRAMFILES}/GLEW/lib
              ${PROJECT_SOURCE_DIR}/src/nvgl/glew/bin
              ${PROJECT_SOURCE_DIR}/src/nvgl/glew/lib
              DOC "The GLEW library")
ELSE (WIN32)
FIND_PATH( GLEW_INCLUDE_PATH GL/glew.h
           /usr/include
           /usr/local/include
           /sw/include
           /opt/local/include
           DOC "The directory where GL/glew.h resides")
FIND_LIBRARY( GLEW_LIBRARY
              NAMES GLEW glew
              PATHS
              /usr/lib64
              /usr/lib
              /usr/local/lib64
              /usr/local/lib
              /sw/lib
              /opt/local/lib
              DOC "The GLEW library")
ENDIF (WIN32)

IF (GLEW_INCLUDE_PATH)
SET( GLEW_FOUND 1 CACHE STRING "Set to 1 if GLEW is found, 0 otherwise")
ELSE (GLEW_INCLUDE_PATH)
SET( GLEW_FOUND 0 CACHE STRING "Set to 1 if GLEW is found, 0 otherwise")
ENDIF (GLEW_INCLUDE_PATH)

MARK_AS_ADVANCED( GLEW_FOUND )

FIND_PACKAGE(GLUT REQUIRED)
FIND_PACKAGE(OpenGL REQUIRED QUIET)
LIST(APPEND LINK_LIBS "${OPENGL_LIBRARIES};${GLEW_LIBRARY};${GLUT_LIBRARY};${OPENGL_LIBRARIES}"
  )
LIST(APPEND LIB_DIRS  "${OPENGL_LIBRARY_DIR};${GLEW_LIBRARY_DIR}" )
LINK_DIRECTORIES(${LIB_DIRS})
LINK_LIBRARIES(${LINK_LIBS})
MESSAGE(STATUS "link dirs: ${LIB_DIRS}")
MESSAGE(STATUS "link libs: ${LINK_LIBS}")

catkin_package(
    CATKIN_DEPENDS
      roscpp
      roslib
      pcl_ros
      cv_bridge
    DEPENDS
      PCL
      Eigen
    INCLUDE_DIRS
      include
    LIBRARIES
      ${PROJECT_NAME}
)

add_library(${PROJECT_NAME}
  src/simulation_io.cpp
  src/camera.cpp
  src/glsl_shader.cpp
  src/model.cpp
  src/range_likelihood.cpp
  src/scene.cpp
  src/sum_reduce.cpp
  src/renderScene.cpp)

target_link_libraries (${PROJECT_NAME} ${Boost_LIBRARIES} ${catkin_LIBRARIES} ${OpenMP_CXX_FLAGS}
                       ${VTK_IO_TARGET_LINK_LIBRARIES}
                       ${GLEW_LIBRARIES} ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES}
                       ${GLEW_LIBRARIES}
                      ${VTK_LIBRARIES}
                       )

install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})


