cmake_minimum_required(VERSION 3.16)
project(Optimizer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Ceres REQUIRED)
find_package(Glog REQUIRED)

add_executable(optimizer optimizer.cc)

target_include_directories(optimizer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(optimizer PRIVATE
    Ceres::ceres
    glog::glog
)