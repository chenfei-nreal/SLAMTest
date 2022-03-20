cmake_minimum_required(VERSION 3.5)

project(eigen_operate)

find_package( Eigen3 REQUIRED )
include_directories( ${EIGEN3_INCLUDE_DIR} )

add_executable(eigen_operate main.cpp)

add_executable(transform transformation.cpp)
