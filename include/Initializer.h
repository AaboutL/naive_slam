//
// Created by hanfuyong on 2022/8/31.
//

#ifndef NAIVE_SLAM_INITIALIZER_H
#define NAIVE_SLAM_INITIALIZER_H

#include "Matcher.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "Map.h"
#include "KeyFrameDB.h"

namespace Naive_SLAM{

class LinkInfo{
public:
    std::vector<int> mvMatchesBetweenKF;
    std::vector<int> mvKPIdToMatchedId; // 每个关键点对应的匹配上的点在新的vector中的下标，如果不存在则为-1
    float mfAverageParallax; // 匹配点之间的平均视差
    std::vector<cv::Point2f> mvPoints1;
    std::vector<cv::Point2f> mvPoints2;
    int mnLinkId;
    int mnMatchedNum;
};

class Initializer{
public:
    Initializer(int nSlidingWindowSize, const cv::Mat& K, const cv::Mat& distCoef,
                Map* pMap, KeyFrameDB* pKeyFrameDB);

    void Insert(KeyFrame* pKF, cv::Mat& img);

    bool Initialize(std::vector<KeyFrame*>& vpKFs, std::set<MapPoint*>& spMPs);
    void CoarseInit(int startId, const LinkInfo& linkInfo);
    int SolveRelativePose(const std::vector<cv::Point2f>& vPts0,
                          const std::vector<cv::Point2f>& vPts1,
                          cv::Mat& R10, cv::Mat& t10);

    cv::Mat Triangulate(const std::vector<cv::Point2f>& vPts0, const std::vector<cv::Point2f>& vPts1,
                     const cv::Mat& R10, const cv::Mat& t10);
    cv::Mat TransformPoints(const cv::Mat& points4D, const cv::Mat& T);

    cv::Point2f project(const cv::Mat &pt3d) const;
    bool ReadyToInit();
    void DealWithUnused();
    void InitBetweenKFs(std::vector<KeyFrame*>& vKFs);
    bool CheckPt3DValid(const cv::Mat& pt3D, const cv::Point2f& ptUn);

private:
    int mnSlidingWindowSize;
    cv::Mat mK;
    cv::Mat mDistCoef;
    std::vector<KeyFrame*> mvpSlidingWindowKFs;
    std::set<MapPoint*> mspSlidingWindowMPs;

    KeyFrame* mpInitKF;
    int mnStartId;
    std::map<int, std::vector<LinkInfo>> mmMatchPairs;
    cv::Mat mStartImg;
    Map* mpMap;
    KeyFrameDB* mpKeyFrameDB;
    std::vector<cv::Mat> mKFImgs;

};
}

#endif //NAIVE_SLAM_INITIALIZER_H
