cmake_minimum_required(VERSION 2.8)
project(naive_slam)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native")

set(OpenCV_DIR "/home/aal/3rdparty/opencv-3.4.16/share/OpenCV")
set(Eigen3_DIR "/home/aal/3rdparty/eigen-3.3.7/share/eigen3/cmake")

find_package(OpenCV 3.4 REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
include_directories(${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR})

FILE(GLOB CPPS ${PROJECT_SOURCE_DIR}/src/*)
message(${CPPS})
add_library(${PROJECT_NAME} SHARED
    ${CPPS})

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS})

add_executable(test test.cc)
target_link_libraries(test ${PROJECT_NAME})


add_executable(test1 test1.cc)