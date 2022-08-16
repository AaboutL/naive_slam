/*
 * @Author: hanfuyong
 * @Date: 2022-07-06 16:09:15
 * @LastEditors: hanfuyong
 * @LastEditTime: 2022-07-27 15:13:38
 * @FilePath: /naive_slam/include/IMU.h
 * @Description: 仅用于个人学习
 * 
 * Copyright (c) 2022 by hanfuyong, All Rights Reserved. 
 */

#pragma once

#include <iostream>
#include <Eigen/Eigen>

namespace Naive_SLAM{

class IMU{
public:
    Eigen::Vector3d ba;
    Eigen::Vector3d bg;
    Eigen::Vector3d acc;
    Eigen::Vector3d gro;
};

class IMUBatch{
public:
    std::vector<IMU> vIMUBatch;
    Eigen::Vector3d mVelocity;
    Eigen::Vector3d mPosition;
    Eigen::Quaterniond mQ;
    Eigen::Matrix3d mR;
public:
    void Integrate();
};

}