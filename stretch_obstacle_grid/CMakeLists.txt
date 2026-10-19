cmake_minimum_required(VERSION 3.10)
project(stretch_obstacle_grid)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
# numpy 2 needs pybind11 >= 2.12, newer than Ubuntu 24.04's 2.11, so prefer a pip-installed
# pybind11 (pip install --user "pybind11>=2.12") over the system one.
execute_process(
  COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE PIP_PYBIND11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
find_package(pybind11 2.12 CONFIG REQUIRED HINTS ${PIP_PYBIND11_DIR})

# The grid library
add_library(${PROJECT_NAME}_core STATIC
  src/cylindrical_grid.cpp
  src/volume_table.cpp
  src/self_model.cpp
  src/lidar.cpp
  src/obstacle_grid.cpp
)
target_include_directories(${PROJECT_NAME}_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME}_core PUBLIC Eigen3::Eigen OpenMP::OpenMP_CXX)

# Python module: import stretch_obstacle_grid
pybind11_add_module(${PROJECT_NAME} src/python_bindings.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}_core)
ament_get_python_install_dir(PYTHON_INSTALL_DIR)
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${PYTHON_INSTALL_DIR})

install(DIRECTORY include/ DESTINATION include)
install(TARGETS ${PROJECT_NAME}_core EXPORT export_${PROJECT_NAME} ARCHIVE DESTINATION lib)
# Debug tools (Python): a bag replay with timing and pictures, and an RViz view.
install(PROGRAMS tools/replay_bag.py tools/grid_viz_node.py tools/calibrate_lidars.py DESTINATION lib/${PROJECT_NAME})
install(FILES tools/grid_setup.py DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY rviz DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_grid test/test_grid.cpp)
  target_link_libraries(test_grid ${PROJECT_NAME}_core)
endif()

ament_export_targets(export_${PROJECT_NAME})
ament_export_dependencies(Eigen3 OpenMP)
ament_package()
