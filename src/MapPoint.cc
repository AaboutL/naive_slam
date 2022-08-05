/*
 * @Author: hanfuyong
 * @Date: 2022-07-30 00:44:25
 * @LastEditors: hanfuyong
 * @LastEditTime: 2022-08-02 14:42:08
 * @FilePath: /naive_slam/src/MapPoint.cc
 * @Description: 仅用于个人学习
 * 
 * Copyright (c) 2022 by hanfuyong, All Rights Reserved. 
 */
#include "MapPoint.h"

namespace Naive_SLAM{
MapPoint::MapPoint(const cv::Point3f& mp, KeyFrame* pRefKF){
    mPoint = mp;
    mpRefKF = pRefKF;
    mvpKFs.emplace_back(pRefKF);
}

void MapPoint::AddKeyFrame(KeyFrame* pKF){
    mvpKFs.emplace_back(pKF);
}
}