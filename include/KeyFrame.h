/*
 * @Author: hanfuyong
 * @Date: 2022-07-27 15:19:46
 * @LastEditors: hanfuyong
 * @LastEditTime: 2022-08-01 22:33:09
 * @FilePath: /naive_slam/include/KeyFrame.h
 * @Description: 仅用于个人学习
 * 
 * Copyright (c) 2022 by hanfuyong, All Rights Reserved. 
 */
#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>

#include "Frame.h"
#include "MapPoint.h"


namespace Naive_SLAM{

class MapPoint;

class KeyFrame{
public:
    KeyFrame(const Frame& frame);
    void AddMapPoints(std::vector<MapPoint*>& mapPoints);

    cv::Mat GetRotation() const;
    cv::Mat GetTranslation() const;
    cv::Mat GetRotationInv() const;
    cv::Mat GetCameraCenter() const;

    void SetRotation(const cv::Mat& Rcw);
    void SetTranslation(const cv::Mat& tcw);

private:
    std::vector<MapPoint*> mvpMapPoints;
    std::vector<cv::KeyPoint> mvKeyPoints;
    std::vector<cv::KeyPoint> mvKeyPointsUn;
    cv::Mat mDescriptions;

    cv::Mat mRcw;
    cv::Mat mtcw;
    cv::Mat mRwc;
    cv::Mat mtwc;
};
}