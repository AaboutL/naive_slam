/*
 * @Author: hanfuyong
 * @Date: 2022-07-05 10:39:47
 * @LastEditors: hanfuyong
 * @LastEditTime: 2022-07-06 11:43:41
 * @FilePath: /naive_slam/include/Frame.h
 * @Description: 仅用于个人学习
 * 
 * Copyright (c) 2022 by hanfuyong, All Rights Reserved. 
 */

#include <iostream>
#include <opencv2/opencv.hpp>

#include "ORBextractor.h"


namespace Naive_SLAM{
class Frame
{
public:
    Frame(const cv::Mat &img, const double& timestamp, ORBextractor* extractor, const cv::Mat& K, const cv::Mat& distCoef);
    void ExtractORB(const cv::Mat& img);


public:
    cv::Mat mK;
    static float fx;
    static float fy;
    static float cx;
    static float cy;
    cv::Mat mDistCoef;
    int N;
    double mTimeStamp;
    
    ORBextractor* mpORBextractor;

private:
    void UndistortKeyPoints();

    std::vector<cv::KeyPoint> mvKeyPoints;
    std::vector<cv::KeyPoint> mvKeyPointsUn;
    cv::Mat mDescriptions;
    cv::Mat mRcw;
    cv::Mat mtcw;
    cv::Mat mRwc;
    cv::Mat mtwc; // == camera center in world frame
};
}