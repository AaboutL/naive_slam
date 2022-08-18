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
#include "tools.h"
#include <chrono>

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
        std::cout << "[Estimator] Param file not exist..." << std::endl;
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

    mpORBExtractorInit = new ORBextractor(500, fs["level_factor"], fs["pyramid_num"], fs["FAST_th_init"], fs["FAST_th_min"]);
    mpORBExtractor = new ORBextractor(fs["feature_num"], fs["level_factor"], fs["pyramid_num"], fs["FAST_th_init"], fs["FAST_th_min"]);

    mImgWidth = fs["Camera.width"];
    mImgHeight = fs["Camera.height"];
    mCellSize = fs["cell_size"];
    mGridCols = (int)std::ceil((float)mImgWidth / (float)mCellSize);
    mGridRows = (int)std::ceil((float)mImgHeight / (float)mCellSize);
}

void Estimator::Estimate(const cv::Mat& image, const double& timestamp){
    mImGray = image;
    if(image.channels() == 3){
        cv::cvtColor(mImGray, mImGray, cv::COLOR_BGR2GRAY);
    }

    if (mState==NO_IMAGE){
        mCurrentFrame = Frame(mImGray, timestamp, mpORBExtractorInit, mK, mDistCoef,
                              mImgWidth, mImgHeight, mCellSize, mGridRows, mGridCols);
        mState = NOT_INITIALIZED;
        mLastFrame = Frame(mCurrentFrame);
        return;
    }
    if (mState == NOT_INITIALIZED){
        mCurrentFrame = Frame(mImGray, timestamp, mpORBExtractorInit, mK, mDistCoef,
                              mImgWidth, mImgHeight, mCellSize, mGridRows, mGridCols);
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
        mCurrentFrame = Frame(mImGray, timestamp, mpORBExtractor, mK, mDistCoef,
                              mImgWidth, mImgHeight, mCellSize, mGridRows, mGridCols);
//        std::chrono::system_clock::time_point t0 = std::chrono::system_clock::now();
//        bool bOK = TrackWithOpticalFlow();
//        std::chrono::system_clock::time_point t1 = std::chrono::system_clock::now();
//        double duration = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
//        std::cout << "KLT cost: " << duration << std::endl;
//        bOK = false;
//        std::cout << "[Estimate: TrackWithOpticalFlow] track state: " << bOK << std::endl;
//        if(!bOK){
            std::chrono::system_clock::time_point t2 = std::chrono::system_clock::now();
            bool bOK = TrackWithKeyFrame();
            std::chrono::system_clock::time_point t3 = std::chrono::system_clock::now();
            double duration1 = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2).count();
            std::cout << "project track cost: " << duration1 << std::endl;
            std::cout << "[Estimate: TrackWithKeyFrame] track state: " << bOK << std::endl;
//        }
    }
    mLastFrame = Frame(mCurrentFrame);
}

bool Estimator::Initialize(){
    auto* initKF = new KeyFrame(mLastFrame);
    auto* curKF = new KeyFrame(mCurrentFrame);

    std::vector<int> matchIdx = SearchByOpticalFlow();
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
//    DrawMatches(mLastFrame.mImg, mCurrentFrame.mImg, ptsMchLast, ptsMchCur,
//                mLastFrame.mvPointsUn, mCurrentFrame.mvPointsUn);

    cv::Mat mask;
    cv::Mat EMat = cv::findEssentialMat(ptsMchLast, ptsMchCur, mK, cv::RANSAC, 0.999, 5.99, mask);
    cv::Mat R, t;
    int inlier_cnt = cv::recoverPose(EMat, ptsMchLast, ptsMchCur, mK, R, t, mask);
    std::cout << "[Initialize] recoverPose inlier num " << inlier_cnt << std::endl;
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
    std::cout << "[Initialize] triangled good 3d points num: " << vpts.size() << std::endl;

    if (vpts.size() < 20)
        return false;

    std::vector<int> matchKPWithMPLast(mLastFrame.mvPoints.size(), -1); // 每个关键点对应的地图点的索引
    std::vector<int> matchKPWithMPCur(mCurrentFrame.mvPoints.size(), -1);
    for(int i = 0; i < vpts.size(); i++){
        cv::Point3f pt = vpts[i];
        int idx = goodMPIdx[i];
        matchKPWithMPLast[matchMPWithKPLast[idx]] = i;
        matchKPWithMPCur[matchMPWithKPCur[idx]] = i;

        cv::Mat description = mLastFrame.mDescriptions.row(matchMPWithKPLast[idx]);
        auto* mapPoint = new MapPoint(pt, mpInitKF);
        mapPoint->SetDescription(description);
        mapPoint->AddKeyFrame(curKF);
        curKF->AddMapPoint(mapPoint);
        initKF->AddMapPoint(mapPoint);
        mpMap->AddMapPoint(mapPoint);
        mspSlidingWindowMPs.insert(mapPoint);
    }

    mLastFrame.SetKeyPointsAndMapPointsMatchIdx(matchKPWithMPLast);
    mCurrentFrame.SetKeyPointsAndMapPointsMatchIdx(matchKPWithMPCur);
    mpInitKF = initKF;
    curKF->SetT(R, t);
    mCurrentFrame.SetT(R, t);

    mpKeyFrameDB->AddKeyFrame(initKF);
    mpKeyFrameDB->AddKeyFrame(curKF);

    mqpSlidingWindowKFs.emplace_back(mpInitKF);
    mqpSlidingWindowKFs.emplace_back(curKF);

    return true;
}

/*
 * 更新运动模型
 */
void Estimator::UpdateVelocity() {
    cv::Mat lastTwc = mLastFrame.GetTwc();
    mVelocity = mCurrentFrame.GetTcw() * lastTwc;
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

bool Estimator::SearchGrid(const cv::Point2f& pt2d, const cv::Mat& description, std::vector<size_t>** grid, float radius, int& matchedId){

    // 找到以这个点为圆心，radius为半径的圆上的所有像素点所在的grid位置，对这些cell中的点遍历搜索。
    int nMinCellX = std::max(0, (int)std::floor(pt2d.x - radius) / mCellSize);
    int nMinCellY = std::max(0, (int)std::floor(pt2d.y - radius) / mCellSize);
    int nMaxCellX = std::min(mGridCols-1, (int)std::ceil(pt2d.x + radius) / mCellSize);
    int nMaxCellY = std::min(mGridRows-1, (int)std::ceil(pt2d.y + radius) / mCellSize);

    int bestDist = INT_MAX;
    int bestId = -1;
    for (int ci = nMinCellY; ci <= nMaxCellY; ci++){
        for (int cj = nMinCellX; cj <= nMaxCellX; cj++){
            std::vector<std::size_t> candidatePtsIdx = grid[ci][cj];
            for (auto j : candidatePtsIdx){
                cv::Mat curDesp = mCurrentFrame.mDescriptions.row(j);
                int dist = DescriptorDistance(description, curDesp);
                if(dist < bestDist){
                    bestDist = dist;
                    bestId = j;
                }
            }
        }
    }
    if(bestDist < 50){
        matchedId = bestId;
        return true;
    }
    return false;
}

std::vector<int> Estimator::SearchByOpticalFlow() {
    cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
    std::vector<uchar> status;
    std::vector<float> err;
    std::vector<cv::Point2f> ptsLK;
    cv::calcOpticalFlowPyrLK(mLastFrame.mImg, mCurrentFrame.mImg, mLastFrame.mvPoints, ptsLK,
                             status, err, cv::Size(21, 21), 1, criteria);
    std::vector<size_t>** grid = mCurrentFrame.GetGrid();
    std::vector<int> matchesIdx(mLastFrame.N, -1);
    int nMatches = 0;
    for (int i = 0; i < ptsLK.size(); i++){
        if(status[i] != 1)
            continue;
        cv::Point2f pt = ptsLK[i];
        if(pt.x >= mImgWidth || pt.x < 0 || pt.y < 0 || pt.y >= mImgHeight)
            continue;
        cv::Mat description = mLastFrame.mDescriptions.row(i);
        int matchedId;
        if(SearchGrid(pt, description, grid, 13, matchedId)){
            matchesIdx[i] = matchedId;
            nMatches++;
        }
    }
    std::cout << "[SearchByOpticalFlow] KL tracking num: " << nMatches << std::endl;
    return matchesIdx;
}

/**
 * 1、用光流法跟踪上一帧的关键点
 * 2、通过索引查找跟踪到的点对应的MapPoints
 * 3、把对应的MapPoints投影到当前帧，通过最小化重投影误差来优化当前帧的位姿。
 * 4、通过优化后的位姿计算当前帧关键点的重投影误差，如果误差小于阈值，则记录这个关键点对应的MapPoints的索引
 */
bool Estimator::TrackWithOpticalFlow() {
    std::vector<int> matchIdx = SearchByOpticalFlow();

    std::vector<int> vLastKPsAndMPsMatch = mLastFrame.GetKeyPointsAndMapPointsMatchIdx();
    std::vector<int> vCurKPsAndMPsMatch(mCurrentFrame.N, -1);

    int numTrackedMP = 0;
    for (size_t i = 0; i < matchIdx.size(); i++){
        int curKeyPointId = matchIdx[i];
        int mapPointId = vLastKPsAndMPsMatch[i];
        if(curKeyPointId == -1) // 上一帧的点是否被光流跟踪到
            continue;
        if(mapPointId == -1) // 上一帧的点是否有对应的MapPoint
            continue;
        vCurKPsAndMPsMatch[curKeyPointId] = mapPointId;
        numTrackedMP++;
    }
    std::cout << "[TrackWithOpticalFlow] num of tracked MapPoints: " << numTrackedMP << std::endl;
    if (numTrackedMP < 20)
        return false;
    mCurrentFrame.SetKeyPointsAndMapPointsMatchIdx(vCurKPsAndMPsMatch);
    std::vector<MapPoint*> mapPoints;
    std::vector<cv::Point2f> pointsUn;
    for(size_t i = 0; i < vCurKPsAndMPsMatch.size(); i++){
        if (vCurKPsAndMPsMatch[i] == -1)
            continue;
        pointsUn.emplace_back(mCurrentFrame.mvPointsUn[i]);
        mapPoints.emplace_back(mpMap->GetMapPoint(vCurKPsAndMPsMatch[i]));
    }

    cv::Mat frameTcw = mVelocity * mLastFrame.GetTcw();
    int nInliers = Optimization::PoseOptimize(pointsUn, mapPoints, mK, frameTcw);
    std::cout << "[TrackWithOpticalFlow] PoseOptimize inliers num: " << nInliers << std::endl;
    if(nInliers > 3){
        mCurrentFrame.SetT(frameTcw);
        return true;
    }
    else
        return false;
}

std::vector<int> Estimator::SearchByProjection(const std::vector<MapPoint*>& mapPoints, const cv::Mat& Tcw) {
    cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
    cv::Mat tcw = Tcw.rowRange(0, 3).col(3);
    std::vector<size_t>** grid = mCurrentFrame.GetGrid();
    std::vector<int> matchesIdx(mapPoints.size(), -1);
    int nMatches = 0;
    for(int i = 0; i < mapPoints.size(); i++){
        MapPoint* pMP = mapPoints[i];
        cv::Point3f pt3dw = pMP->GetWorldPos();
        cv::Mat matDesc = pMP->GetDescription();
        cv::Mat pt_tmp(cv::Matx<float, 3, 1>(pt3dw.x, pt3dw.y, pt3dw.z));
        cv::Mat pt3dc = Rcw * pt_tmp + tcw;
        cv::Point2f pt2dUn = project(pt3dc);
        if (pt2dUn.x >= mImgWidth || pt2dUn.x < 0 || pt2dUn.y < 0 || pt2dUn.y >= mImgHeight)
            continue;
        int matchedId;
        if(SearchGrid(pt2dUn, matDesc, grid, 40, matchedId)){
            matchesIdx[i] = matchedId;
            nMatches++;
        }
    }
    std::cout << "[SearchByProjection] MapPoint matched nums: " << nMatches << std::endl;
    return matchesIdx;
}

bool Estimator::TrackWithKeyFrame() {
    KeyFrame* lastestKF = mqpSlidingWindowKFs.back();
    std::vector<MapPoint*> mapPointsInKF = lastestKF->GetMapPoints();
    cv::Mat frameTcw = mVelocity * mLastFrame.GetTcw();
    std::vector<int> matchesIdx = SearchByProjection(mapPointsInKF, frameTcw);
    std::vector<MapPoint*> mapPoints;
    std::vector<cv::Point2f> pointsUn;
    for(int i = 0; i < matchesIdx.size(); i++){
        if(matchesIdx[i] == -1)
            continue;
        mapPoints.emplace_back(mapPointsInKF[i]);
        pointsUn.emplace_back(mCurrentFrame.mvPointsUn[matchesIdx[i]]);
    }

    int nInliers = Optimization::PoseOptimize(pointsUn, mapPoints, mK, frameTcw);
    if(nInliers > 3){
        mCurrentFrame.SetT(frameTcw);
        return true;
    }
    else
        return false;
}

cv::Point2f Estimator::project(const cv::Mat &pt3d) const {
    float x_norm = pt3d.at<float>(0) / pt3d.at<float>(2);
    float y_norm = pt3d.at<float>(1) / pt3d.at<float>(2);
    float x_un = x_norm * fx + cx;
    float y_un = y_norm * fy + cy;
    return {x_un, y_un};
}

}
