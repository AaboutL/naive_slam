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
#include "Optimization.h"
#include <queue>

namespace Naive_SLAM{

void DrawMatches(const cv::Mat& img1, const cv::Mat& img2, const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2){
    int w = img1.size().width;
    int h = img1.size().height;
    cv::Mat imgShow(h, w * 2, CV_8UC3, cv::Scalar::all(0));
    cv::Mat tmp;
    cv::cvtColor(img1, tmp, cv::COLOR_GRAY2BGR);
    tmp.copyTo(imgShow(cv::Rect(0, 0, w, h)));
    cv::cvtColor(img2, tmp, cv::COLOR_GRAY2BGR);
    tmp.copyTo(imgShow(cv::Rect(w, 0, w, h)));
    cv::resize(imgShow, imgShow, imgShow.size() * 2);
    for (size_t i = 0; i < points1.size(); i++){
        cv::circle(imgShow, points1[i] * 2, 3, cv::Scalar(255, 0, 0));
        cv::circle(imgShow, (points2[i] + cv::Point2f(w, 0)) * 2, 3, cv::Scalar(0, 255, 0));
        cv::line(imgShow, points1[i] * 2, (points2[i] + cv::Point2f(w, 0)) * 2, cv::Scalar(255, 0, 0));
    }
    cv::imshow("match", imgShow);
    cv::waitKey(0);
}

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
    mDistCoef.at<float>(0, 0) = fs["Camera.k1"];
    mDistCoef.at<float>(1, 0) = fs["Camera.k2"];
    mDistCoef.at<float>(2, 0) = fs["Camera.p1"];
    mDistCoef.at<float>(3, 0) = fs["Camera.p2"];

    mpORBExtractorInit = new ORBextractor(300, fs["level_factor"], fs["pyramid_num"], fs["FAST_th_init"], fs["FAST_th_min"]);
    mpORBExtractor = new ORBextractor(fs["feature_num"], fs["level_factor"], fs["pyramid_num"], fs["FAST_th_init"], fs["FAST_th_min"]);

    mCellSize = fs["cell_size"];
}

void Estimator::Estimate(const cv::Mat& image, const double& timestamp){
    mImGray = image;
    if(image.channels() == 3){
        cv::cvtColor(mImGray, mImGray, cv::COLOR_BGR2GRAY);
    }

    if (mState==NO_IMAGE){
        mCurrentFrame = Frame(mImGray, timestamp, mpORBExtractorInit, mK, mDistCoef, mCellSize);
        mState = NOT_INITIALIZED;
        mLastFrame = Frame(mCurrentFrame);
        return;
    }
    if (mState == NOT_INITIALIZED){
        mCurrentFrame = Frame(mImGray, timestamp, mpORBExtractorInit, mK, mDistCoef, mCellSize);
        bool flag = Initialize();
        if(flag){
            mLastFrame = Frame(mCurrentFrame);
            UpdateVelocity();
            mState = OK;
        }
        else{
            mState = NO_IMAGE;
        }
        return;
    }
    if(mState == OK){
        mCurrentFrame = Frame(mImGray, timestamp, mpORBExtractor, mK, mDistCoef, mCellSize);
        bool bOK = TrackWithOpticalFlow();
        std::cout << "track state: " << bOK << std::endl;
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

std::vector<int> Estimator::SearchInArea(const std::vector<cv::Point2f>& ptsLK, const std::vector<uchar>& status){
    auto curGrid = mCurrentFrame.GetGrid();
    std::vector<int> matchIdx(mLastFrame.N, -1);
    for (int i = 0; i < mLastFrame.N; i++){
        if(status[i] != 1)
            continue;
        cv::Point2f pt = ptsLK[i];
        if(pt.x >= mImGray.cols || pt.x < 0 || pt.y < 0 || pt.y >= mImGray.rows)
            continue;
        int colIdx = pt.x / mCellSize;
        int rowIdx = pt.y / mCellSize;
        std::vector<std::size_t> candidatePtsIdx = curGrid[rowIdx][colIdx];
        cv::Mat lastDesp = mLastFrame.mDescriptions.row(i);
        int bestDist = INT_MAX;
        int bestId = -1;
        for (auto j : candidatePtsIdx){
            cv::Mat curDesp = mCurrentFrame.mDescriptions.row(j);
            int dist = DescriptorDistance(lastDesp, curDesp);
            if(dist < bestDist){
                bestDist = dist;
                bestId = j;
            }
        }
        if(bestDist < 40){
            matchIdx[i] = bestId;
        }
    }
    return matchIdx;
}

bool Estimator::Initialize(){
    auto* initKF = new KeyFrame(mLastFrame);
    auto* curKF = new KeyFrame(mCurrentFrame);
    cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
    std::vector<uchar> status;
    std::vector<float> err;
    std::vector<cv::Point2f> ptsLK;
    cv::calcOpticalFlowPyrLK(mLastFrame.mImg, mCurrentFrame.mImg, mLastFrame.mvPoints, ptsLK, status, err, cv::Size(21, 21), 8, criteria);
    std::vector<int> matchIdx = SearchInArea(ptsLK, status);
    std::vector<cv::Point2f> ptsMchLast, ptsMchCur;
    std::vector<int> matchMPWithKPLast, matchMPWithKPCur; // 剔除不好的地图点前，每个地图点对应的关键点的索引
    for (size_t i = 0; i < matchIdx.size(); i++){
        if (matchIdx[i] == -1)
            continue;
        ptsMchLast.emplace_back(mLastFrame.mvPointsUn[mLastFrame.mvL0KPIndices[i]]);
        ptsMchCur.emplace_back(mCurrentFrame.mvPointsUn[matchIdx[i]]);
        matchMPWithKPLast.push_back(i);
        matchMPWithKPCur.push_back(matchIdx[i]);

    }
    cv::Mat mask;
    cv::Mat EMat = cv::findEssentialMat(ptsMchLast, ptsMchCur, mK, cv::RANSAC, 0.999, 3.0, mask);
    cv::Mat R, t;
    int inlier_cnt = cv::recoverPose(EMat, ptsMchLast, ptsMchCur, mK, R, t, mask);
    if (inlier_cnt < 8)
        return false;

    cv::Mat P1(3, 4, CV_32F, cv::Scalar(0));
    mK.copyTo(P1.rowRange(0, 3).colRange(0, 3));
    cv::Mat P2(3, 4, CV_32F, cv::Scalar(0));
    R.copyTo(P2.rowRange(0, 3).colRange(0, 3));
    t.copyTo(P2.rowRange(0, 3).col(3));
    R.convertTo(R, CV_32F);
    t.convertTo(t, CV_32F);
    P2 = mK * P2;
    cv::Mat points4D;
    cv::triangulatePoints(P1, P2, ptsMchLast, ptsMchCur, points4D);

    std::vector<cv::Point3f> vpts;
    std::vector<int> goodMPIdx; // 剔除不好的地图点之后的索引与之前的索引的关系
    for (int i = 0; i < points4D.cols; i++){
        cv::Mat p3dC1 = points4D.col(i);
        p3dC1 = p3dC1.rowRange(0, 3) / p3dC1.at<float>(3);
        if(p3dC1.at<float>(2) <= 0) {
            matchMPWithKPLast[i] = -1;
            matchMPWithKPCur[i] = -1;
            continue;
        }
        cv::Mat p3dC2 = R * p3dC1 + t;
        if(p3dC2.at<float>(2) <= 0) {
            matchMPWithKPLast[i] = -1;
            matchMPWithKPCur[i] = -1;
            continue;
        }

        cv::Point3f pt = cv::Point3f(p3dC1.at<float>(0), p3dC1.at<float>(1), p3dC1.at<float>(2));
        vpts.emplace_back(pt);
        goodMPIdx.push_back(i);
    }

    if (vpts.size() < 20)
        return false;

    std::vector<int> matchKPWithMPLast(mLastFrame.mvPoints.size(), -1); // 每个关键点对应的地图点的索引
    std::vector<int> matchKPWithMPCur(mCurrentFrame.mvPoints.size(), -1);
    for(int i = 0; i < vpts.size(); i++){
        cv::Point3f pt = vpts[i];
        int idx = goodMPIdx[i];
        matchKPWithMPLast[matchMPWithKPLast[idx]] = i;
        matchKPWithMPCur[matchMPWithKPCur[idx]] = i;
        auto* mapPoint = new MapPoint(pt, mpInitKF);
        mapPoint->AddKeyFrame(curKF);
        curKF->AddMapPoint(mapPoint);
        initKF->AddMapPoint(mapPoint);
        mpMap->AddMapPoint(mapPoint);
    }

    mLastFrame.SetKeyPointsAndMapPointsMatchIdx(matchKPWithMPLast);
    mCurrentFrame.SetKeyPointsAndMapPointsMatchIdx(matchKPWithMPCur);
    mpInitKF = initKF;
    curKF->SetT(R, t);
    mCurrentFrame.SetT(R, t);

    mpKeyFrameDB->AddKeyFrame(initKF);
    mpKeyFrameDB->AddKeyFrame(curKF);

    return true;
}

void Estimator::UpdateVelocity() {
    cv::Mat lastTwc = mLastFrame.GetTwc();
    mVelocity = mCurrentFrame.GetTcw() * lastTwc;
}

/**
 * 1、用光流法跟踪上一帧的关键点
 * 2、通过索引查找跟踪到的点对应的MapPoints
 * 3、把对应的MapPoints投影到当前帧，通过最小化重投影误差来优化当前帧的位姿。
 * 4、通过优化后的位姿计算当前帧关键点的重投影误差，如果误差小于阈值，则记录这个关键点对应的MapPoints的索引
 */
bool Estimator::TrackWithOpticalFlow() {
    cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
    std::vector<uchar> status;
    std::vector<float> err;
    std::vector<cv::Point2f> ptsLK;
    cv::calcOpticalFlowPyrLK(mLastFrame.mImg, mCurrentFrame.mImg, mLastFrame.mvPoints, ptsLK, status, err, cv::Size(21, 21), 8, criteria);
    std::vector<int> matchIdx = SearchInArea(ptsLK, status); // matchIdx记录上一帧每个关键点对应的当前帧关键点的索引
    std::vector<int> vLastKPsAndMPsMatch = mLastFrame.GetKeyPointsAndMapPointsMatchIdx();
    std::vector<int> vCurKPsAndMPsMatch(mCurrentFrame.N, -1);
    for (size_t i = 0; i < matchIdx.size(); i++){
        int curKeyPointId = matchIdx[i];
        int mapPointId = vLastKPsAndMPsMatch[i];
        if(curKeyPointId == -1) // 上一帧的点是否被光流跟踪到
            continue;
        if(mapPointId == -1) // 上一帧的点是否有对应的MapPoint
            continue;
        vCurKPsAndMPsMatch[curKeyPointId] = mapPointId;
    }
    mCurrentFrame.SetKeyPointsAndMapPointsMatchIdx(vCurKPsAndMPsMatch);
    cv::Mat Tcw_opt;
    if(PoseOptimization(vCurKPsAndMPsMatch, Tcw_opt)){
        mCurrentFrame.SetT(Tcw_opt);
        return true;
    }
    else
        return false;
}

/*
 * 1. 获取当前帧匹配的2D特征点和3D地图点，通过最小化重投影误差方法优化当前帧的位姿。
 * 2. 根据优化后的位姿，把2D-3D匹配点中，大于阈值的匹配关系去掉。
 */
bool Estimator::PoseOptimization(const std::vector<int>& curKPsAndMPsMatch, cv::Mat& Tcw) {
    Tcw = mVelocity * mLastFrame.GetTcw();
    std::vector<MapPoint*> mapPoints;
    std::vector<cv::Point2f> pointsUn;
    for(size_t i = 0; i < curKPsAndMPsMatch.size(); i++){
        if (curKPsAndMPsMatch[i] == -1)
            continue;
        pointsUn.emplace_back(mCurrentFrame.mvPointsUn[i]);
        mapPoints.emplace_back(mpMap->GetMapPoint(curKPsAndMPsMatch[i]));
    }
    int nInliers = Optimization::PoseOptimize(pointsUn, mapPoints, mK, Tcw);
    if(nInliers < 3)
        return false;
    return true;
}

}
