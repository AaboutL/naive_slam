cmake_minimum_required(VERSION 2.8)
project(naive_slam)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native -std=c++14")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0028 NEW)
set(G2O_ROOT "/home/aal/3rdparty/g2o")
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
FILE(GLOB G2O_LIBS ${G2O_ROOT}/lib/*.so)

#set(Eigen3_DIR "/home/aal/3rdparty/eigen-3.3.7/share/eigen3/cmake")
set(OpenCV_DIR "/home/aal/3rdparty/opencv-3.4.16/share/OpenCV")

#set(Pangolin_DIR "/home/aal/software/Pangolin/build")
# set(Pangolin_DIR "/home/aal/3rdparty/Pangolin/lib/cmake/Pangolin")
find_package(Eigen3 REQUIRED)
find_package(OpenCV 3.4 REQUIRED)
find_package(G2O REQUIRED)
include_directories(${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${G2O_INCLUDE_DIR}
        )

FILE(GLOB CPPS ${PROJECT_SOURCE_DIR}/src/*)
add_library(${PROJECT_NAME} SHARED
    ${CPPS})

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
#    ${G2O_CORE_LIBRARY}
#    ${G2O_TYPES_SLAM3D}
#    ${G2O_SOLVER_DENSE}
#    ${G2O_TYPES_SBA}
#    ${G2O_STUFF_LIBRARY}
    ${G2O_LIBS}
    glog
    )

add_executable(main main.cc)
target_link_libraries(main ${PROJECT_NAME})

#add_executable(test test.cc)
#target_link_libraries(test ${PROJECT_NAME})
#
#add_executable(test1 test1.cc)
#target_link_libraries(test1 ${OpenCV_LIBS})
#
#add_executable(test_pangolin test_pangolin.cc)
#target_link_libraries(test_pangolin ${PROJECT_NAME})
#
#add_executable(test_pangolin_euroc test_pangolin_euroc.cc)
#target_link_libraries(test_pangolin_euroc ${PROJECT_NAME})