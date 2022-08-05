/*
 * @Author: hanfuyong
 * @Date: 2022-07-27 15:41:10
 * @LastEditors: hanfuyong
 * @LastEditTime: 2022-07-30 02:01:51
 * @FilePath: /naive_slam/include/MapPoint.h
 * @Description: 仅用于个人学习
 * 
 * Copyright (c) 2022 by hanfuyong, All Rights Reserved. 
 */
#pragma once

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "KeyFrame.h"

namespace Naive_SLAM{

class KeyFrame;

class MapPoint{
public:
    MapPoint(const cv::Point3f& mp, KeyFrame* pRefKF);
    void AddKeyFrame(KeyFrame* pKF);

private:
    std::vector<std::pair<KeyFrame*, int>> mvObservations;
    std::vector<KeyFrame*> mvpKFs;
    KeyFrame* mpRefKF;
    cv::Point3f mPoint;
    cv::Mat mDescriptor;

};

}