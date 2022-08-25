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
    MapPoint(const MapPoint& mapPoint);
    MapPoint(const cv::Mat& mp, KeyFrame* pRefKF);
    void AddKeyFrame(KeyFrame* pKF);
    cv::Mat GetWorldPos() const;
    void SetWorldPos(const cv::Mat& worldPos);
    void SetDescription(const cv::Mat& description);
    cv::Mat GetDescription() const;

    void AddObservation(KeyFrame* pKF, int id);
    void EraseObservation(KeyFrame* pKF);
    int GetIdxInKF(KeyFrame* pKF);
    int GetObsNum() const;

private:
    std::map<KeyFrame*, int> mmObservations; // 观察到此mappoint的关键帧，以及对应关键点的索引
    std::vector<KeyFrame*> mvpKFs;
    KeyFrame* mpRefKF;
    cv::Mat mWorldPos;
    cv::Mat mDescription;

};

}