/*
 * @Author: hanfuyong
 * @Date: 2022-07-27 15:19:53
 * @LastEditors: hanfuyong
 * @LastEditTime: 2022-07-30 01:09:38
 * @FilePath: /naive_slam/src/KeyFrame.cc
 * @Description: 仅用于个人学习
 * 
 * Copyright (c) 2022 by hanfuyong, All Rights Reserved. 
 */
#include "KeyFrame.h"

namespace Naive_SLAM{
KeyFrame::KeyFrame(const Frame& frame){
    mvKeyPoints = frame.mvKeyPoints;
    mvKeyPointsUn = frame.mvKeyPointsUn;
    mDescriptions = frame.mDescriptions;
    mRcw = frame.GetRotation().clone();
    mtcw = frame.GetTranslation().clone();
    mTcw = frame.GetTcw();
    mRwc = frame.GetRotationInv().clone();
    mtwc = frame.GetCameraCenter().clone();
    mTwc = frame.GetTwc();
}

void KeyFrame::AddMapPoint(MapPoint* mapPoint){
//    mvpMapPoints.emplace_back(mapPoint);
    mvpMapPoints.push_back(mapPoint);
}

cv::Mat KeyFrame::GetRotation() const {
    return mRcw;
}

cv::Mat KeyFrame::GetTranslation() const {
    return mtcw;
}

cv::Mat KeyFrame::GetRotationInv() const {
    return mRwc;
}

cv::Mat KeyFrame::GetCameraCenter() const {
    return mtwc;
}

void KeyFrame::SetRotation(const cv::Mat& Rcw){
    mRcw = Rcw;
    mRwc = mRcw.t();
}
void KeyFrame::SetTranslation(const cv::Mat& tcw){
    mtcw = tcw;
    mtwc = -mRwc * mtcw;
}
void KeyFrame::SetT(const cv::Mat& Rcw, const cv::Mat& tcw){
    mRcw = Rcw;
    mtcw = tcw;
    mTcw = cv::Mat::eye(4, 4, CV_32F);
    mRcw.copyTo(mTcw.rowRange(0, 3).colRange(0, 3));
    mtcw.copyTo(mTcw.rowRange(0, 3).col(3));

    mTwc = cv::Mat::eye(4, 4, CV_32F);
    mRwc = Rcw.t();
    mtwc = -Rcw.t() * tcw;
    mRwc.copyTo(mTwc.rowRange(0, 3).colRange(0, 3));
    mtwc.copyTo(mTwc.rowRange(0, 3).col(3));
}

void KeyFrame::SetT(const cv::Mat& Tcw){
    mTcw = Tcw;

    mRcw = Tcw.rowRange(0, 3).colRange(0, 3);
    mtcw = Tcw.rowRange(0, 3).col(3);
    mTwc = cv::Mat::eye(4, 4, CV_32F);
    mRwc = mRcw.t();
    mtwc = -mRwc * mtcw;

    mRwc.copyTo(mTwc.rowRange(0, 3).colRange(0, 3));
    mtwc.copyTo(mTwc.rowRange(0, 3).col(3));
}

}