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
#include <DBoW2/FORB.h>
#include <DBoW2/TemplatedVocabulary.h>

#include "Frame.h"
#include "MapPoint.h"


namespace Naive_SLAM{

class MapPoint;

class KeyFrame{
public:
    explicit KeyFrame(const Frame& frame);
    void AddMapPoint(int id, MapPoint* mapPoint);

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

    std::vector<MapPoint*> GetMapPoints() const;
    MapPoint* GetMapPoint(int id) const;

    std::vector<cv::Point2f> GetPoints() const;
    std::vector<cv::Point2f> GetPointsUn() const;
    std::vector<cv::Point2f> GetPointsLevel0() const;
    std::vector<cv::Point2f> GetPointsUnLevel0() const;
    cv::KeyPoint GetKeyPoint(int id) const;
    cv::KeyPoint GetKeyPointUn(int id) const;
    cv::Mat GetDescription(int id) const;

    void SetMatchKPWithMP(const std::vector<int>& matchKPWithMP);
    std::vector<int> GetMatchKPWithMP() const;
    void ComputeBow();
    DBoW2::BowVector GetBowVec() const;
    DBoW2::FeatureVector GetFeatVec() const;

    void EraseMapPoint(MapPoint* pMP);
    void SetMapPoints(const std::vector<MapPoint*>& vpMPs);
    std::vector<float> GetScaleFactors() const;
    std::vector<float> GetLevelSigma2() const;
    std::vector<float> GetInvLevelSigma2() const;

    cv::Mat ComputeFundamental(KeyFrame* pKF);

public:
    int N;

private:
    cv::Mat mK;
    cv::Mat mDistCoef;
    std::vector<MapPoint*> mvpMapPoints;
    std::vector<cv::KeyPoint> mvKeyPoints;
    std::vector<cv::KeyPoint> mvKeyPointsUn;
    std::vector<int> mvMatchKPWithMP;
    cv::Mat mDescriptions;

    cv::Mat mRcw;
    cv::Mat mtcw;
    cv::Mat mTcw;
    cv::Mat mRwc;
    cv::Mat mtwc;
    cv::Mat mTwc;

    Vocabulary *mpORBvocabulary;
    DBoW2::BowVector mBowVector;
    DBoW2::FeatureVector mFeatVector;

    std::vector<float> mvScaleFactors;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

public:
    int mImgWidth;
    int mImgHeight;
    int mCellSize;
    int mGridRows;
    int mGridCols;
    std::vector<std::size_t>** mGrid;
};
}