cmake_minimum_required(VERSION 3.5)

project(useOpenCV)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(PkgConfig)
pkg_check_modules(YAMLCPP REQUIRED yaml-cpp>=0.5)
include_directories(${YAMLCPP_INCLUDE_DIRS})


add_executable(useOpenCV main.cpp)
target_link_libraries(useOpenCV ${OpenCV_LIBS} ${YAMLCPP_LIBRARIES})
