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

KeyFrame::KeyFrame(const Frame& frame):mnId(frame.mnId), N(frame.N), mK(frame.mK),
mDistCoef(frame.mDistCoef),mvKeyPoints(frame.mvKeyPoints),
mvKeyPointsUn(frame.mvKeyPointsUn),mDescriptions(frame.mDescriptions.clone()),
mRcw(frame.GetRotation().clone()), mtcw(frame.GetTranslation().clone()),
mTcw(frame.GetTcw().clone()), mRwc(frame.GetRotationInv().clone()),
mtwc(frame.GetCameraCenter().clone()), mTwc(frame.GetTwc().clone()),
mvScaleFactors(frame.mvScaleFactors), mvLevelSigma2(frame.mvLevelSigma2),
mvInvLevelSigma2(frame.mvInvLevelSigma2),
mImgWidth(frame.mImgWidth), mImgHeight(frame.mImgHeight), mCellSize(frame.mCellSize),
mGridRows(frame.mGridRows), mGridCols(frame.mGridCols), mGrid(frame.mGrid){
    mpORBVocabulary = frame.mpORBVocabulary;
    mvpMapPoints.resize(frame.N, nullptr);
    ComputeBow();
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

std::vector<cv::Point2f> KeyFrame::GetPointsUn() const {
    std::vector<cv::Point2f> points(mvKeyPointsUn.size());
    for (size_t i = 0; i < mvKeyPointsUn.size(); i++){
        points[i] = mvKeyPointsUn[i].pt;
    }
    return points;
}

std::vector<cv::Point2f> KeyFrame::GetPointsLevel0() const {
    std::vector<cv::Point2f> points;
    points.reserve(mvKeyPoints.size() / 2);
    for (const auto & mvKeyPoint : mvKeyPoints){
        if(mvKeyPoint.octave==0) {
            points.emplace_back(mvKeyPoint.pt);
        }
    }
    return points;
}

std::vector<cv::Point2f> KeyFrame::GetPointsUnLevel0() const {
    std::vector<cv::Point2f> points;
    points.reserve(mvKeyPointsUn.size() / 2);
    for (const auto & mvKeyPoint : mvKeyPointsUn){
        if(mvKeyPoint.octave==0) {
            points.emplace_back(mvKeyPoint.pt);
        }
    }
    return points;
}

cv::KeyPoint KeyFrame::GetKeyPoint(int id) const {
    return mvKeyPoints[id];
}

cv::KeyPoint KeyFrame::GetKeyPointUn(int id) const {
    return mvKeyPointsUn[id];
}

std::vector<cv::KeyPoint> KeyFrame::GetKeyPointsUn() const {
    return mvKeyPointsUn;
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
        mpORBVocabulary->transform(vDesc, mBowVector, mFeatVector, 4);
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

std::vector<float> KeyFrame::GetScaleFactors() const {
    return mvScaleFactors;
}

std::vector<float> KeyFrame::GetLevelSigma2() const {
    return mvLevelSigma2;
}

std::vector<float> KeyFrame::GetInvLevelSigma2() const {
    return mvInvLevelSigma2;
}

cv::Mat KeyFrame::ComputeFundamental(KeyFrame *pKF) {
    cv::Mat R2w = pKF->GetRotation();
    cv::Mat t2w = pKF->GetTranslation();
    cv::Mat R12 = mRcw * R2w.t();
    cv::Mat t12 = -R12 * t2w + mtcw;
    cv::Mat t12_skew = cv::Mat(cv::Matx33f(0, -t12.at<float>(2, 0), t12.at<float>(1, 0),
                                           t12.at<float>(2, 0), 0, -t12.at<float>(0, 0),
                                           t12.at<float>(2, 0), t12.at<float>(0, 0), 0));
    return mK.t().inv() * t12_skew * R12 * mK.inv();
}

int KeyFrame::GetMapPointNum() const {
    int nMPNum = 0;
    for(int i = 0; i < N; i++){
        MapPoint* pMP = mvpMapPoints[i];
        if(pMP && !pMP->IsBad())
            nMPNum++;
    }
    return nMPNum;
}

}