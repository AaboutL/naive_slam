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

#include <iostream>
#include <vector>

#include "Frame.h"
#include "KeyFrame.h"
#include "IMU.h"
#include "ORBextractor.h"
#include "MapPoint.h"
#include "Map.h"
#include "KeyFrameDB.h"
#include "Vocabulary.h"

namespace Naive_SLAM{

class Estimator{
    
public:
    Estimator(float fx, float fy, float cx, float cy, float k1, float k2, float p1, float p2);
    Estimator(const std::string& strParamFile, Map* pMap, KeyFrameDB* pKeyFrameDB,
              Vocabulary *pORBVocabulary);
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

    int mImgWidth;
    int mImgHeight;
    int mCellSize;
    int mGridRows;
    int mGridCols;
    cv::Mat mVelocity;

    std::vector<KeyFrame*> mvpSlidingWindowKFs;
    std::set<MapPoint*> mspSlidingWindowMPs;

    cv::Mat mLastestKFImg;

    int mnKeyFrameMatchInliers;
    int mnSlidingWindowMatchInliers;
    Vocabulary *mpORBVocabulary;

    int mSlidingWindowSize;

    std::vector<MapPoint*> mvpCurrentTrackedMPs;

private:
    bool Initialize();

    int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

    bool SearchGrid(const cv::Point2f& pt2d, const cv::Mat& description, std::vector<size_t>** grid, float radius,
                    int& matchedId);
    std::vector<int> SearchByOpticalFlow();
    bool TrackWithOpticalFlow();
    void UpdateVelocity();
    bool TrackWithKeyFrame();
    std::vector<int> SearchByProjection(const std::vector<MapPoint*>& mapPoints, const cv::Mat& Tcw);
    int SearchByProjection(std::vector<MapPoint*>& vMapPoints, std::vector<cv::Point2f>& vPointsUn,
                           std::vector<int>& vMatchedIdx);
    cv::Point2f project(const cv::Mat& pt3d) const;
    bool TrackWithinSlidingWindow();

    bool NeedNewKeyFrame();
    KeyFrame* CreateKeyFrame();
    void CreateNewMapPoints(KeyFrame* pKF);
    cv::Mat ComputeF12(KeyFrame* pKF1, KeyFrame* pKF2);
    bool CheckDistEpipolarLine(const cv::KeyPoint& kp1, const cv::KeyPoint& kp2, const cv::Mat& F12);

    void SlidingWindowBA();

    void Marginalize();
};
}