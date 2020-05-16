project(OpenGR-PCLWrapper)
add_compile_options(-std=c++11 -O3 -fopenmp)

file(GLOB_RECURSE Demo_PCL_headers
                  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/*.h
                  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/*.hpp)

find_package(PCL REQUIRED)
find_package(OpenMP REQUIRED)

message("PCL_LIBRARIES "  ${PCL_LIBRARIES})

if( PCL_FOUND )
  message (STATUS "[Demos] PCL found, creating target OpenGR-PCLWrapper")
  set(Demo_PCL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp )

  # [C/C++]>[General]>[Additional Include Directories]
  include_directories( ${PCL_INCLUDE_DIRS} )

  # [C/C++]>[Preprocessor]>[Preprocessor Definitions]
  add_definitions( ${PCL_DEFINITIONS} )

  include_directories( ${CMAKE_CURRENT_SOURCE_DIR}  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/)
  add_executable(${PROJECT_NAME} ${Demo_PCL_SRC} ${Demo_PCL_headers})
  target_link_libraries(${PROJECT_NAME} opengr_algo opengr_utils ${PCL_LIBRARIES} ${OpenMP_CXX_FLAGS})

  

  # For Use Not PreCompiled Features
  #add_definitions( -DPCL_NO_PRECOMPILE )

  # [Linker]>[General]>[Additional Library Directories]
  # link_directories( ${PCL_LIBRARY_DIRS} )

  install( TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)

endif()


install( DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
         DESTINATION include
         FILES_MATCHING REGEX ".*(h|hpp)$")
