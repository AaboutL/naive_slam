/*
 * @Author: hanfuyong
 * @Date: 2022-07-27 15:10:17
 * @LastEditors: hanfuyong
 * @LastEditTime: 2022-08-02 19:05:54
 * @FilePath: /naive_slam/include/Estimator.h
 * @Description: 仅用于个人学习
 * 
 * Copyright (c) 2022 by hanfuyong, All Rights Reserved. 
 */
#pragma once

#include "Frame.h"
#include "KeyFrame.h"
#include "IMU.h"
#include "ORBextractor.h"
#include "MapPoint.h"
#include "Map.h"
#include "KeyFrameDB.h"

#include <iostream>
#include <vector>
#include <deque>

namespace Naive_SLAM{

class Estimator{
    
public:
    Estimator(float fx, float fy, float cx, float cy, float k1, float k2, float p1, float p2);
    Estimator(const std::string& strParamFile, Map* pMap, KeyFrameDB* pKeyFrameDB);
    enum State{
        NO_IMAGE = 0,
        NOT_INITIALIZED = 1,
        OK = 2
    };


    void Estimate(const cv::Mat& image, const double& timestamp);

    cv::Mat mK;
    float fx;
    float fy;
    float cx;
    float cy;
    cv::Mat mDistCoef;

    cv::Mat mRwc;
    cv::Mat mtwc;

private:
    State mState;

    cv::Mat mImGray;

    std::deque<KeyFrame*> mqSlidingWindow;
    Frame mCurrentFrame;
    Frame mLastFrame;

    KeyFrame* mpInitKF;

    ORBextractor* mpORBExtractor;
    ORBextractor* mpORBExtractorInit;

    KeyFrameDB* mpKeyFrameDB;
    Map* mpMap;

    int mCellSize;
    cv::Mat mVelocity;
    
private:
    bool Initialize();

    int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
    std::vector<int> SearchInArea(const std::vector<cv::Point2f>& ptsLK, const std::vector<uchar>& status);
    void TrackWithOpticalFlow();
    void UpdateVelocity();
};
}