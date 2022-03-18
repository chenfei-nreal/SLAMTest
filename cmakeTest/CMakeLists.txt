cmake_minimum_required(VERSION 3.5)

project(hello_library)

# static library
add_library(hello_library STATIC src/hello.cpp)
target_include_directories(hello_library PUBLIC ${PROJECT_SOURCE_DIR}/include)

# dynamic library
add_library(hello_library_so SHARED src/hello.cpp)
target_include_directories(hello_library_so PUBLIC ${PROJECT_SOURCE_DIR}/include)

# test
add_executable(hello_static src/main.cpp)
target_link_libraries(hello_static PRIVATE hello_library)

add_executable(hello_dynamic src/main.cpp)
target_link_libraries(hello_dynamic PRIVATE hello_library_so)
