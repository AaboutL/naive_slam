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

Estimator::Estimator(const std::string& strParamFile, Map* pMap, KeyFrameDB* pKeyFrameDB,
                     Vocabulary *pORBVocabulary):
mpKeyFrameDB(pKeyFrameDB), mpMap(pMap), mpORBVocabulary(pORBVocabulary){
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
    mSlidingWindowSize = fs["SlidingWindowSize"];
}

void Estimator::Estimate(const cv::Mat& image, const double& timestamp){
    mImGray = image;
    if(image.channels() == 3){
        cv::cvtColor(mImGray, mImGray, cv::COLOR_BGR2GRAY);
    }

    if (mState==NO_IMAGE){
        mCurrentFrame = Frame(mImGray, timestamp, mpORBExtractorInit, mK, mDistCoef,
                              mImgWidth, mImgHeight, mCellSize, mGridRows, mGridCols, mpORBVocabulary);
        mState = NOT_INITIALIZED;
        mLastFrame = Frame(mCurrentFrame);
        return;
    }
    if (mState == NOT_INITIALIZED){
        mCurrentFrame = Frame(mImGray, timestamp, mpORBExtractorInit, mK, mDistCoef,
                              mImgWidth, mImgHeight, mCellSize, mGridRows, mGridCols, mpORBVocabulary);
        bool flag = Initialize();
        if(flag){
            mLastFrame = Frame(mCurrentFrame);
            UpdateVelocity();
            mLastestKFImg = mImGray;
            mState = OK;
        }
        else{
            mState = NO_IMAGE;
        }
        return;
    }
    if(mState == OK){
        mCurrentFrame = Frame(mImGray, timestamp, mpORBExtractor, mK, mDistCoef,
                              mImgWidth, mImgHeight, mCellSize, mGridRows, mGridCols, mpORBVocabulary);
        std::chrono::system_clock::time_point t2 = std::chrono::system_clock::now();
        bool bOK = TrackWithKeyFrame();
        std::chrono::system_clock::time_point t3 = std::chrono::system_clock::now();
        double duration1 = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2).count();
        std::cout << "project track cost: " << duration1 << std::endl;
        std::cout << "[Estimate: TrackWithKeyFrame] track state: " << bOK << std::endl;

        if (bOK){
            bOK = TrackWithinSlidingWindow();
        }

        if(bOK){
            if(NeedNewKeyFrame()){
                KeyFrame* pNewKF = CreateKeyFrame();
                // 新关键帧与滑窗中老的关键帧之间创建新的地图点
                CreateNewMapPoints(pNewKF);
                SlidingWindowBA();
                if(mvpSlidingWindowKFs.size() < mSlidingWindowSize){
                    mvpSlidingWindowKFs.emplace_back(pNewKF);
                    // 同时，把新关键帧对应的mappoint插入到滑窗中
                    for(auto* pMP : pNewKF->GetMapPoints()){
                        if(pMP)
                            mspSlidingWindowMPs.insert(pMP);
                    }
                }
                else{
                    // 从slidingWindow中删除旧的关键帧和地图点, 并插入新的关键帧。
                    Marginalize();
                    mvpSlidingWindowKFs.emplace_back(pNewKF);
                    // 更新地图点的滑窗，先清空，然后重新插入。
                    mspSlidingWindowMPs.clear();
                    for(auto* pKF: mvpSlidingWindowKFs){
                        auto vpMPs = pKF->GetMapPoints();
                        for(auto * pMP : vpMPs){
                            if(pMP)
                                mspSlidingWindowMPs.insert(pMP);
                        }
                    }
                }
            }
        }

        UpdateVelocity();
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

    std::vector<cv::Mat> vpts;
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

//        cv::Point3f pt = cv::Point3f(p3dC1.at<float>(0), p3dC1.at<float>(1), p3dC1.at<float>(2));
        vpts.emplace_back(p3dC1);
        goodMPIdx.push_back(i);
    }
    std::cout << "[Initialize] triangled good 3d points num: " << vpts.size() << std::endl;

    if (vpts.size() < 20)
        return false;

    std::vector<int> matchKPWithMPLast(mLastFrame.mvPoints.size(), -1); // 每个关键点对应的地图点的索引
    std::vector<int> matchKPWithMPCur(mCurrentFrame.mvPoints.size(), -1);
    for(int i = 0; i < vpts.size(); i++){
        cv::Mat pt = vpts[i];
        int idx = goodMPIdx[i];
        matchKPWithMPLast[matchMPWithKPLast[idx]] = i;
        matchKPWithMPCur[matchMPWithKPCur[idx]] = i;

        cv::Mat description = mLastFrame.mDescriptions.row(matchMPWithKPLast[idx]);
        auto* mapPoint = new MapPoint(pt, mpInitKF);
        mapPoint->SetDescription(description);
//        mapPoint->AddKeyFrame(curKF);
        mapPoint->AddObservation(initKF, matchMPWithKPLast[idx]);
        initKF->AddMapPoint(matchMPWithKPLast[idx], mapPoint);

        mapPoint->AddObservation(curKF, matchMPWithKPCur[idx]);
        curKF->AddMapPoint(matchMPWithKPCur[idx], mapPoint);
        mpMap->AddMapPoint(mapPoint);
        mspSlidingWindowMPs.insert(mapPoint);
    }

    mLastFrame.SetKeyPointsAndMapPointsMatchIdx(matchKPWithMPLast);
    mCurrentFrame.SetKeyPointsAndMapPointsMatchIdx(matchKPWithMPCur);
    mpInitKF = initKF;
    mpInitKF->SetMatchKPWithMP(matchMPWithKPLast);
    curKF->SetT(R, t);
    curKF->SetMatchKPWithMP(matchKPWithMPCur);
    mCurrentFrame.SetT(R, t);

    mpKeyFrameDB->AddKeyFrame(initKF);
    mpKeyFrameDB->AddKeyFrame(curKF);

    mvpSlidingWindowKFs.emplace_back(mpInitKF);
    mvpSlidingWindowKFs.emplace_back(curKF);

    return true;
}

/*
 * 更新运动模型
 */
void Estimator::UpdateVelocity() {
    cv::Mat lastTwc = mLastFrame.GetTwc();
    PrintMat("lastTwc: ", lastTwc);
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
//    KeyFrame* lastestKF = mqpSlidingWindowKFs.back();
//    std::vector<cv::Point2f> lastestKFPoints = lastestKF->GetPoints();

    cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
    std::vector<uchar> status;
    std::vector<float> err;
    std::vector<cv::Point2f> ptsLK;
    cv::calcOpticalFlowPyrLK(mLastFrame.mImg, mCurrentFrame.mImg, mLastFrame.mvPoints, ptsLK,
                             status, err, cv::Size(21, 21), 1, criteria);
//    cv::calcOpticalFlowPyrLK(mLastestKFImg, mCurrentFrame.mImg, lastestKFPoints, ptsLK,
//                             status, err, cv::Size(21, 21), 1, criteria);
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
        if(SearchGrid(pt, description, grid, 40, matchedId)){
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
    std::vector<bool> bOutlier;
    int nInliers = Optimization::PoseOptimize(pointsUn, mapPoints, mK, frameTcw, bOutlier);
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
        cv::Mat pt3dw = pMP->GetWorldPos();
        cv::Mat matDesc = pMP->GetDescription();
        cv::Mat pt3dc = Rcw * pt3dw + tcw;
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
    KeyFrame* lastestKF = mvpSlidingWindowKFs.back();
    std::vector<MapPoint*> mapPointsInKF = lastestKF->GetMapPoints();
    cv::Mat frameTcw = mVelocity * mLastFrame.GetTcw();
    PrintMat("Velocity Tcw: ", frameTcw);

    std::vector<int> matchesIdx = SearchByProjection(mapPointsInKF, frameTcw);
    std::vector<MapPoint*> mapPoints;
    std::vector<cv::Point2f> pointsUnMatched;
    for(int i = 0; i < matchesIdx.size(); i++){
        if(matchesIdx[i] == -1)
            continue;
        mapPoints.emplace_back(mapPointsInKF[i]);
        pointsUnMatched.emplace_back(mCurrentFrame.mvPointsUn[matchesIdx[i]]);
    }
    std::vector<bool> bOutlier;
    int nInliers = Optimization::PoseOptimize(pointsUnMatched, mapPoints, mK, frameTcw, bOutlier);

    // 画图
    DrawPoints(mCurrentFrame, pointsUnMatched, bOutlier);

    if(nInliers > 10){
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

int Estimator::SearchByProjection(std::vector<MapPoint*>& vMapPoints, std::vector<cv::Point2f>& vPointsUn,
                                  std::vector<int>& vMatchedIdx) {
    vMatchedIdx.resize(mCurrentFrame.N, -1);
    cv::Mat Tcw = mCurrentFrame.GetTcw();
    cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
    cv::Mat tcw = Tcw.rowRange(0, 3).col(3);
    std::vector<size_t>** grid = mCurrentFrame.GetGrid();
    int nMatches = 0;
    for(const auto& pMP: mspSlidingWindowMPs){
        cv::Mat pt3dw = pMP->GetWorldPos();
        cv::Mat matDesc = pMP->GetDescription();
        cv::Mat pt3dc = Rcw * pt3dw + tcw;
        cv::Point2f pt2dUn = project(pt3dc);
        if (pt2dUn.x >= mImgWidth || pt2dUn.x < 0 || pt2dUn.y < 0 || pt2dUn.y >= mImgHeight)
            continue;

        int matchedId;
        if(SearchGrid(pt2dUn, matDesc, grid, 40, matchedId)){
            vPointsUn.emplace_back(mCurrentFrame.mvPointsUn[matchedId]);
            vMapPoints.emplace_back(pMP);
            vMatchedIdx[matchedId] = nMatches;
            nMatches++;
        }
    }
    return nMatches;
}

bool Estimator::TrackWithinSlidingWindow() {
    std::vector<MapPoint*> vMapPointsMatched;
    std::vector<cv::Point2f> vPointsUnMatched;
    std::vector<int> vMatchedIdx;
    int nMatches = SearchByProjection(vMapPointsMatched, vPointsUnMatched, vMatchedIdx);

    cv::Mat frameTcw = mCurrentFrame.GetTcw();
    std::vector<bool> bOutlier;
    mnMatchInliers = Optimization::PoseOptimize(vPointsUnMatched, vMapPointsMatched, mK, frameTcw, bOutlier);

    // 画图
    DrawPoints(mCurrentFrame, vPointsUnMatched, bOutlier);
    if(mnMatchInliers > 30){
        mCurrentFrame.SetT(frameTcw);
        mvpCurrentTrackedMPs.clear();
        mvpCurrentTrackedMPs.resize(mCurrentFrame.N, nullptr);
        for(int i = 0; i < mCurrentFrame.N; i++){
            if(!bOutlier[vMatchedIdx[i]]){
                mvpCurrentTrackedMPs[i] = vMapPointsMatched[vMatchedIdx[i]];
            }
        }

        return true;
    }
    else
        return false;
}

bool Estimator::NeedNewKeyFrame() {
    float thLastestKFMatch = 0.8;
    int nLastestKFMapPoints = static_cast<int>(mvpSlidingWindowKFs.back()->GetPoints().size());
    if (mnMatchInliers > 15 && mnMatchInliers < thLastestKFMatch * nLastestKFMapPoints){
        return true;
    }
    return false;
}

KeyFrame* Estimator::CreateKeyFrame() {
    auto *pCurKF = new KeyFrame(mCurrentFrame);
    pCurKF->SetMapPoints(mvpCurrentTrackedMPs);
    pCurKF->ComputeBow();
    return pCurKF;
}

void Estimator::CreateNewMapPoints(KeyFrame* pKF) {
    for (auto* pKFSW : mvpSlidingWindowKFs) {
        pKFSW->ComputeBow();
        const DBoW2::FeatureVector &vFeatVec1 = pKF->GetFeatVec();
        const DBoW2::FeatureVector &vFeatVec2 = pKFSW->GetFeatVec();
        DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
        DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
        DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
        DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

        cv::Mat F12 = ComputeF12(pKF, pKFSW);

        std::vector<bool> vbMatched2(pKFSW->N, false);
        std::vector<int> vMatched12(pKF->N, -1);
        while (f1it != f1end || f2it != f2end) {
            if (f1it == f2it) {
                for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++) {
                    const size_t idx1 = f1it->second[i1];
                    MapPoint *pMP1 = pKF->GetMapPoint(i1);
                    if (pMP1)
                        continue;
                    cv::KeyPoint kp1Un = pKF->GetKeyPointUn(i1);
                    cv::Mat description1 = pKF->GetDescription(i1);

                    int bestDist = 50;
                    int bestIdx2 = 0;
                    for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2; i2++) {
                        const size_t idx2 = f2it->second[i2];
                        MapPoint *pMP2 = pKFSW->GetMapPoint(i2);
                        if (vbMatched2[idx2] || pMP2)
                            continue;
                        cv::KeyPoint kp2Un = pKFSW->GetKeyPointUn(i2);
                        cv::Mat description2 = pKFSW->GetDescription(i2);
                        int dist = DescriptorDistance(description1, description2);
                        if (dist > bestDist)
                            continue;

                        if (CheckDistEpipolarLine(kp1Un, kp2Un, F12)) {
                            bestIdx2 = idx2;
                            bestDist = dist;
                        }
                    }
                    if (bestIdx2 >= 0) {
                        cv::KeyPoint kp2 = pKFSW->GetKeyPointUn(bestIdx2);
                        vMatched12[idx1] = bestIdx2;
                        vbMatched2[bestIdx2] = true;
                    }
                }
                f1it++;
                f2it++;
            } else if (f1it->first < f2it->first) {
                f1it = vFeatVec1.lower_bound(f2it->first);
            } else {
                f2it = vFeatVec2.lower_bound(f1it->first);
            }
        }

        // Collect Matched points
        std::vector<std::pair<int, int>> vIdxMatch12;
        std::vector<cv::Point2f> vPtsMatch1, vPtsMatch2;
        for (int i = 0; i < vMatched12.size(); i++) {
            if (vMatched12[i] == -1)
                continue;
            vPtsMatch1.emplace_back(pKF->GetKeyPointUn(i).pt);
            vPtsMatch2.emplace_back(pKFSW->GetKeyPointUn(vMatched12[i]).pt);
            vIdxMatch12.emplace_back(std::make_pair(i, vMatched12[i]));
        }
        cv::Mat R1w = pKF->GetRotation();
        cv::Mat t1w = pKF->GetTranslation();
        cv::Mat R2w = pKFSW->GetRotation();
        cv::Mat t2w = pKFSW->GetTranslation();
        // 计算pKF1作为currentKF，pKF2作为滑窗中的KF
        // 重建在pKF1相机位姿下的空间点，然后再转换到世界坐标系下
        // 首先要得到pKF2到pKF1的旋转平移
        cv::Mat R21 = R2w * R1w.t();
        cv::Mat t21 = -R21 * t1w + t2w;
        cv::Mat P1(3, 4, CV_32F, cv::Scalar(0));
        mK.copyTo(P1.rowRange(0, 3).colRange(0, 3));
        cv::Mat P2(3, 4, CV_32F, cv::Scalar(0));
        R21.copyTo(P2.rowRange(0, 3).colRange(0, 3));
        t21.copyTo(P2.rowRange(0, 3).col(3));
        P2 = mK * P2;
        cv::Mat points4D;
        cv::triangulatePoints(P1, P2, vPtsMatch1, vPtsMatch2, points4D);

        for (int i = 0; i < points4D.cols; i++) {
            cv::Mat p3dC1 = points4D.col(i);
            p3dC1 = p3dC1.rowRange(0, 3) / p3dC1.at<float>(3);
            if (p3dC1.at<float>(2) <= 0) {
                continue;
            }
            cv::Mat p3dC2 = R21 * p3dC1 + t21;
            if (p3dC2.at<float>(2) <= 0) {
                continue;
            }

            int matchIdx1 = vIdxMatch12[i].first;
            int matchIdx2 = vIdxMatch12[i].second;
            cv::Mat description = pKF->GetDescription(matchIdx1);
            auto *mapPoint = new MapPoint(p3dC1, pKF);
            mapPoint->SetDescription(description);
            mapPoint->AddObservation(pKF, matchIdx1);
            mapPoint->AddObservation(pKFSW, matchIdx2);
            mpMap->AddMapPoint(mapPoint);

            pKF->AddMapPoint(matchIdx1, mapPoint);
            pKFSW->AddMapPoint(matchIdx2, mapPoint);
            mspSlidingWindowMPs.insert(mapPoint);
        }
    }

}

cv::Mat Estimator::ComputeF12(KeyFrame *pKF1, KeyFrame *pKF2) {
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();
    cv::Mat R12 = R1w * R2w.t();
    cv::Mat t12 = -R12 * t2w + t1w;
    cv::Mat t12_skew = cv::Mat(cv::Matx33f(0, -t12.at<float>(2, 0), t12.at<float>(1, 0),
                                           t12.at<float>(2, 0), 0, -t12.at<float>(0, 0),
                                           t12.at<float>(2, 0), t12.at<float>(0, 0), 0));

    return mK.t().inv() * t12_skew * R12 * mK.inv();
}

bool Estimator::CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12) {
    float a = kp1.pt.x * F12.at<float>(0, 0) + kp1.pt.y * F12.at<float>(0, 1) + F12.at<float>(0, 2);
    float b = kp1.pt.x * F12.at<float>(1, 0) + kp1.pt.y * F12.at<float>(1, 1) + F12.at<float>(1, 2);
    float c = kp1.pt.x * F12.at<float>(2, 0) + kp1.pt.y * F12.at<float>(2, 1) + F12.at<float>(2, 2);

    float num = a * kp2.pt.x + b * kp2.pt.y + c;
    float den = a * a + b * b;
    if (den ==0)
        return false;
    float distSqr = num * num / den;
    return distSqr < 3.84;
}

void Estimator::SlidingWindowBA() {
    Optimization::SlidingWindowBA(mvpSlidingWindowKFs, mK);
}

void Estimator::Marginalize() {
    auto it = mvpSlidingWindowKFs.begin();
    mvpSlidingWindowKFs.erase(it);
}

}
