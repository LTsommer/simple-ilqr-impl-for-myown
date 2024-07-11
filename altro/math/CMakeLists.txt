cmake_minimum_required(VERSION 3.24)

project(geometry_lib)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

set(HEADER
        math_utils.h
        aabox2d.h
        box2d.h
        line_segment2d.h
        polygon2d.h)

set(SOURCE
        math_utils.cc
        aabox2d.cc
        box2d.cc
        line_segment2d.cc
        polygon2d.cc)
add_compile_options(-O2)

add_library(${PROJECT_NAME} ${HEADER} ${SOURCE})
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)

