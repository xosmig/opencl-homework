cmake_minimum_required(VERSION 3.8)
project(opencllab02 C CXX)

set(CMAKE_CXX_STANDARD 17)

#if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_GNUCC OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_DEBUG "-D_GLIBCXX_DEBUG")
#endif()

find_package(OpenCL REQUIRED)
include_directories(${OPENCL_INCLUDE_DIR})

include_directories(src)
file(GLOB SOURCE_FILES "src/*.cpp" "src/*.cl" "src/*.hpp" "src/*.h")
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} OpenCL)
