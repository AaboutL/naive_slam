cmake_minimum_required(VERSION 2.8)
project(naive_slam)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

set(OpenCV_DIR "/home/aal/3rdparty/opencv-3.4.16/share/OpenCV")
set(Eigen3_DIR "/home/aal/3rdparty/eigen-3.3.7/share/eigen3/cmake")
set(Pangolin_DIR "/home/aal/software/Pangolin/build")
# set(Pangolin_DIR "/home/aal/3rdparty/Pangolin/lib/cmake/Pangolin")

find_package(OpenCV 3.4 REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(Pangolin REQUIRED)
include_directories(${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS})

FILE(GLOB CPPS ${PROJECT_SOURCE_DIR}/src/*)
add_library(${PROJECT_NAME} SHARED
    ${CPPS})

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${Pangolin_LIBRARIES})
message(${Pangolin_LIBRARIES})

add_executable(main main.cc)
target_link_libraries(main ${PROJECT_NAME})

add_executable(test test.cc)
target_link_libraries(test ${PROJECT_NAME})

add_executable(test1 test1.cc)

add_executable(test_pangolin test_pangolin.cc)
target_link_libraries(test_pangolin ${PROJECT_NAME})

add_executable(test_pangolin_euroc test_pangolin_euroc.cc)
target_link_libraries(test_pangolin_euroc ${PROJECT_NAME})