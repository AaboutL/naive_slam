/*
 * @Author: hanfuyong
 * @Date: 2022-07-27 15:10:24
 * @LastEditors: hanfuyong
 * @LastEditTime: 2022-08-02 19:06:27
 * @FilePath: /naive_slam/src/Estimator.cc
 * @Description: 仅用于个人学习
 * 
 * Copyright (c) 2022 by hanfuyong, All Rights Reserved. 
 */
#include "Estimator.h"
#include <queue>

namespace Naive_SLAM{
Estimator::Estimator(float fx, float fy, float cx, float cy, float k1, float k2, float p1, float p2):
fx(fx), fy(fy), cx(cx), cy(cy), mState(NO_IMAGE){
    mK = cv::Mat::eye(3, 3, CV_32FC1);
    mK.at<float>(0, 0) = fx;
    mK.at<float>(1, 1) = fy;
    mK.at<float>(0, 2) = cx;
    mK.at<float>(1, 2) = cy;
    mDistCoef = cv::Mat::zeros(4, 1, CV_32FC1);
    mDistCoef.at<float>(0, 0) = k1;
    mDistCoef.at<float>(1, 0) = k2;
    mDistCoef.at<float>(2, 0) = p1;
    mDistCoef.at<float>(3, 0) = p2;
    
    mpORBExtractor = new ORBextractor(300, 1.2, 1, 20, 7);
    mpORBExtractorInit = new ORBextractor(500, 1.2, 1, 20, 7);
}

Estimator::Estimator(const std::string& strParamFile, Map* pMap, KeyFrameDB* pKeyFrameDB):
mpMap(pMap), mpKeyFrameDB(pKeyFrameDB){
    cv::FileStorage fs(strParamFile.c_str(), cv::FileStorage::READ);
    if (!fs.isOpened()){
        std::cout << "Param file not exist..." << std::endl;
        exit(0);
    }
    fx = fs["Camera.fx"];
    fy = fs["Camera.fy"];
    cx = fs["Camera.cx"];
    cy = fs["Camera.cy"];
    mK = cv::Mat::eye(3, 3, CV_32FC1);
    mK.at<float>(0, 0) = fx;
    mK.at<float>(1, 1) = fy;
    mK.at<float>(0, 2) = cx;
    mK.at<float>(1, 2) = cy;

    mDistCoef = cv::Mat::zeros(4, 1, CV_32FC1);
    mDistCoef.at<float>(0, 0) = fs["Camera.k1"];;
    mDistCoef.at<float>(1, 0) = fs["Camera.k2"];
    mDistCoef.at<float>(2, 0) = fs["Camera.p1"];
    mDistCoef.at<float>(3, 0) = fs["Camera.p2"];

    mpORBExtractorInit = new ORBextractor(1000, fs["level_factor"], fs["pyramid_num"], fs["FAST_th_init"], fs["FAST_th_min"]);
    mpORBExtractor = new ORBextractor(fs["feature_num"], fs["level_factor"], fs["pyramid_num"], fs["FAST_th_init"], fs["FAST_th_min"]);
}

void Estimator::Estimate(const cv::Mat& image, const double& timestamp){
    mImGray = image;
    if(image.channels() == 3){
        cv::cvtColor(mImGray, mImGray, cv::COLOR_BGR2GRAY);
    }

    if (mState==NO_IMAGE){
        mCurrentFrame = Frame(mImGray, timestamp, mpORBExtractorInit, mK, mDistCoef);
        // mpInitKF = new KeyFrame(mCurrentFrame);
        mState = NOT_INITIALIZED;
        mLastFrame = Frame(mCurrentFrame);
        return;
    }
    if (mState == NOT_INITIALIZED){
        mCurrentFrame = Frame(mImGray, timestamp, mpORBExtractorInit, mK, mDistCoef);
        bool flag = Initialize();
        if(flag){
            mLastFrame = Frame(mCurrentFrame);
            mState = OK;
        }
        else{
            mState = NO_IMAGE;
        }
        return;
    }
    if(mState == OK){
    }
    mLastFrame = Frame(mCurrentFrame);
}

int Estimator::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

std::vector<int> Estimator::SearchInArea(const std::vector<cv::Point2f>& ptsLK, const std::vector<uchar>& status,
const int cellSize, const cv::Size& imgSize){
    int kpNum = mLastFrame.mvL0KPIndices.size();
    std::vector<int> matchIdx(kpNum, -1);
    for (size_t i = 0; i < kpNum; i++){
        if (!status[i]) 
            continue;
        cv::Point2f pt = ptsLK[i];
        int minCellX = std::max(0, (int)pt.x - cellSize);
        int maxCellX = std::min(imgSize.width, (int)pt.x + cellSize);
        int minCellY = std::max(0, (int)pt.y - cellSize);
        int maxCellY = std::min(imgSize.height, (int)pt.y + cellSize);
        // std::cout << pt << "  " << minCellX <<  " " << maxCellX << " " << minCellY << " " << maxCellY << std::endl;
        int bestDist = INT_MAX;
        int bestId = -1;
        for (size_t j = 0; j < mCurrentFrame.mvKeyPoints.size(); j++){
            cv::KeyPoint kp = mCurrentFrame.mvKeyPoints[j];
            if (kp.octave != 0)
                continue;
            if (kp.pt.x < minCellX || kp.pt.x > maxCellX || kp.pt.y < minCellY || kp.pt.y > maxCellY)
                continue;
            cv::Mat desp = mLastFrame.mDescriptions.row(mLastFrame.mvL0KPIndices[i]);
            cv::Mat desp1 = mCurrentFrame.mDescriptions.row(j);
            int dist = DescriptorDistance(desp, desp1);
            if(dist < bestDist){
                bestDist = dist;
                bestId = j;
            }
        }
        if (bestDist < 40){
            matchIdx[i] = bestId;
        }
    }
    return matchIdx;
}

bool Estimator::Initialize(){
    auto* initKF = new KeyFrame(mLastFrame);
    auto* curKF = new KeyFrame(mCurrentFrame);
    mpKeyFrameDB->AddKeyFrame(initKF);
    mpKeyFrameDB->AddKeyFrame(curKF);
    std::vector<cv::Point2f> pts, pts1;
    std::vector<int> indices;
    cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
    std::vector<uchar> status;
    std::vector<float> err;
    std::vector<cv::Point2f> ptsLK;
    cv::calcOpticalFlowPyrLK(mLastFrame.mImg, mCurrentFrame.mImg, mLastFrame.mvPoints, ptsLK, status, err, cv::Size(21, 21), 8, criteria);
    std::vector<int> matchIdx = SearchInArea(ptsLK, status, 10, mCurrentFrame.mImg.size());
    std::vector<cv::Point2f> ptsMchLast, ptsMchCur;
    std::vector<int> vMPIdxLast(mLastFrame.mvPoints.size(), 0);
    std::vector<int> vMPIdxCur(mCurrentFrame.mvPoints.size(), 0);
    for (size_t i = 0; i < matchIdx.size(); i++){
        if (matchIdx[i] == -1)
            continue;
        ptsMchLast.emplace_back(mLastFrame.mvPointsUn[mLastFrame.mvL0KPIndices[i]]);
        ptsMchCur.emplace_back(mCurrentFrame.mvPointsUn[matchIdx[i]]);

        // ptsMchLast和ptsMchCur用来三角化, vMPIdxLast, vMPIdxCur用来记录前后两帧中关键点对应的地图点
        vMPIdxLast.emplace_back(mLastFrame.mvL0KPIndices[i]);
        vMPIdxCur.emplace_back(matchIdx[i]);
    }
    cv::Mat mask;
    cv::Mat EMat = cv::findEssentialMat(ptsMchLast, ptsMchCur, mK, cv::RANSAC, 0.999, 1.0, mask);
    cv::Mat R, t;
    int inlier_cnt = cv::recoverPose(EMat, ptsMchLast, ptsMchCur, mK, R, t, mask);
    if (inlier_cnt < 8)
        return false;
    cv::Mat P1(3, 4, CV_32F, cv::Scalar(0));
    mK.copyTo(P1.rowRange(0, 3).colRange(0, 3));
    cv::Mat P2(3, 4, CV_32F, cv::Scalar(0));
    R.copyTo(P2.rowRange(0, 3).colRange(0, 3));
    t.copyTo(P2.rowRange(0, 3).col(3));
    P2 = mK * P2;
    cv::Mat points4D;
    cv::triangulatePoints(P1, P2, ptsMchLast, ptsMchCur, points4D);

    std::vector<MapPoint*> pMapPoints;
    for (int i = 0; i < points4D.cols; i++){
        cv::Mat p3dC1 = points4D.col(i).rowRange(0, 3) / points4D.col(i).row(3);
        if(p3dC1.at<float>(2) <= 0)
            continue;
        cv::Mat p3dC2 = R * p3dC1 + t;
        if(p3dC2.at<float>(2) <= 0)
            continue;

        cv::Point3f mp = cv::Point3f(p3dC1.at<float>(0), p3dC1.at<float>(1), p3dC1.at<float>(2));
        auto* mapPoint = new MapPoint(mp, mpInitKF);
        mapPoint->AddKeyFrame(curKF);
        pMapPoints.emplace_back(mapPoint);
        mpMap->AddMapPoint(mapPoint);
    }
    if (pMapPoints.size() < 20)
        return false;
    mpInitKF->AddMapPoints(pMapPoints);
    curKF->AddMapPoints(pMapPoints);
    curKF->mRcw = R;
    curKF->mtcw = t;
    curKF->mRwc = R.t();
    curKF->mtwc = -R.t() * t;
    mCurrentFrame.mRcw = R;
    mCurrentFrame.mtcw= t;
    mCurrentFrame.mRwc = R.t();
    mCurrentFrame.mtwc= -R.t() * t;

    return true;
}

}
