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
#include "Converter.h"

namespace Naive_SLAM{
KeyFrame::KeyFrame(const Frame& frame):N(frame.N){
    mvKeyPoints = frame.mvKeyPoints;
    mvKeyPointsUn = frame.mvKeyPointsUn;
    mDescriptions = frame.mDescriptions;
    mRcw = frame.GetRotation().clone();
    mtcw = frame.GetTranslation().clone();
    mTcw = frame.GetTcw();
    mRwc = frame.GetRotationInv().clone();
    mtwc = frame.GetCameraCenter().clone();
    mTwc = frame.GetTwc();
    mpORBvocabulary = frame.mpORBVocabulary;
    mvpMapPoints.resize(frame.N, nullptr);
}

void KeyFrame::AddMapPoint(int id, MapPoint* mapPoint){
//    mvpMapPoints.emplace_back(mapPoint);
    mvpMapPoints[id] = mapPoint;
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

cv::Mat KeyFrame::GetTcw() const{
    return mTcw;
}

cv::Mat KeyFrame::GetTwc() const{
    return mTwc;
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

std::vector<MapPoint*> KeyFrame::GetMapPoints() const {
    return mvpMapPoints;
}

MapPoint* KeyFrame::GetMapPoint(int id) const {
    return mvpMapPoints[id];
}

std::vector<cv::Point2f> KeyFrame::GetPoints() const {
    std::vector<cv::Point2f> points(mvKeyPoints.size());
    for (size_t i = 0; i < mvKeyPoints.size(); i++){
        points[i] = mvKeyPoints[i].pt;
    }
    return points;
}

cv::KeyPoint KeyFrame::GetKeyPointUn(int id) const {
    return mvKeyPointsUn[id];
}

cv::Mat KeyFrame::GetDescription(int id) const {
    return mDescriptions.row(id);
}

void KeyFrame::SetMatchKPWithMP(const std::vector<int> &matchKPWithMP) {
    mvMatchKPWithMP = matchKPWithMP;
}

std::vector<int> KeyFrame::GetMatchKPWithMP() const {
    return mvMatchKPWithMP;
}

void KeyFrame::ComputeBow() {
    if(mBowVector.empty() || mFeatVector.empty()){
        std::vector<cv::Mat> vDesc = Converter::DescriptionMatToVector(mDescriptions);
        mpORBvocabulary->transform(vDesc, mBowVector, mFeatVector, 4);
    }
}

DBoW2::BowVector KeyFrame::GetBowVec() const {
    return mBowVector;
}

DBoW2::FeatureVector KeyFrame::GetFeatVec() const {
    return mFeatVector;
}

void KeyFrame::EraseMapPoint(MapPoint *pMP) {
    int idx = pMP->GetIdxInKF(this);
    if(idx >= 0){
        mvpMapPoints[idx] = nullptr;
    }
}

void KeyFrame::SetMapPoints(const vector<MapPoint *> &vpMPs) {
    mvpMapPoints = vpMPs;
}

}