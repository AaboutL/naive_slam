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
    mRcw = frame.GetRotation();
    mtcw = frame.GetTranslation();
    mRwc = frame.GetRotationInv();
    mtwc = frame.GetCameraCenter();
}

void KeyFrame::AddMapPoints(std::vector<MapPoint*>& mapPoints){
    mvpMapPoints = mapPoints;
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
}