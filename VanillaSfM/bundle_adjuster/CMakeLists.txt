cmake_minimum_required(VERSION 3.10)
project(BundleAdjuster)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Ceres REQUIRED)
find_package(Glog REQUIRED)

add_executable(bundle_adjuster bundle_adjuster.cc)

target_include_directories(bundle_adjuster PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(bundle_adjuster
  PRIVATE
  Ceres::ceres
  glog::glog)