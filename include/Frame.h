/*
 * @Author: hanfuyong
 * @Date: 2022-07-05 10:39:47
 * @LastEditors: hanfuyong
 * @LastEditTime: 2022-07-27 23:21:37
 * @FilePath: /naive_slam/include/Frame.h
 * @Description: 仅用于个人学习
 * 
 * Copyright (c) 2022 by hanfuyong, All Rights Reserved. 
 */

#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>

#include "ORBextractor.h"
//#include "MapPoint.h"

namespace Naive_SLAM{

//class MapPoint;

class Frame
{

public:
    Frame(){}
    Frame(const Frame& frame);
    Frame(const cv::Mat &img, const double& timestamp, ORBextractor* extractor, const cv::Mat& K, const cv::Mat& distCoef,
        const int cell_size);
    void ExtractORB(const cv::Mat& img);


public:
    int N;
    double mTimeStamp;
    
    ORBextractor* mpORBextractor;

    std::vector<cv::KeyPoint> mvKeyPoints;
    std::vector<cv::KeyPoint> mvKeyPointsUn;
    std::vector<cv::Point2f> mvPoints;
    std::vector<cv::Point2f> mvPointsUn;
    std::vector<int> mvL0KPIndices;
    std::vector<int> mvMapPointIndices; // 每一个关键点对应的地图点的索引
    cv::Mat mDescriptions;

    cv::Mat mImg;

    cv::Mat mK;
    static float fx;
    static float fy;
    static float cx;
    static float cy;
    cv::Mat mDistCoef;

public:
    std::vector<std::vector<std::vector<std::size_t>>> GetGrid();
    void SetKeyPointsAndMapPointsMatchIdx(const std::vector<int>& mapPointsIdx);
    std::vector<int> GetKeyPointsAndMapPointsMatchIdx() const;

    cv::Mat GetRotation() const;
    cv::Mat GetTranslation() const;
    cv::Mat GetRotationInv() const;
    cv::Mat GetCameraCenter() const;
    cv::Mat GetTcw() const;
    cv::Mat GetTwc() const;

    void SetRotation(const cv::Mat& Rcw);
    void SetTranslation(const cv::Mat& tcw);
    void SetT(const cv::Mat& Tcw);
    void SetT(const cv::Mat& Rcw, const cv::Mat& tcw);

private:
    void UndistortKeyPoints();
    void AssignGrid();

private:
    bool mflag;
    cv::Mat mRcw;
    cv::Mat mtcw;
    cv::Mat mTcw;
    cv::Mat mRwc;
    cv::Mat mtwc; // == camera center in world frame
    cv::Mat mTwc;

    int mImgWidth;
    int mImgHeight;
    int mCellSize;
    int mGridRowNum;
    int mGridColNum;
    std::vector<std::vector<std::vector<std::size_t>>> mGrid;

//    std::vector<MapPoint*> mvpMapPoints;

};

}
