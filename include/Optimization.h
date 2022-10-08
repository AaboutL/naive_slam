//
// Created by hanfuyong on 2022/8/10.
//

#ifndef NAIVE_SLAM_OPTIMIZATION_H
#define NAIVE_SLAM_OPTIMIZATION_H

#include "Frame.h"
#include "MapPoint.h"
#include "KeyFrame.h"

namespace Naive_SLAM{

class Optimization{
public:
    static int PoseOptimize(const std::vector<cv::KeyPoint>& vKPsUn, std::vector<MapPoint*>& mapPoints,
                             const std::vector<float>& vInvLevelSigma2,
                             const cv::Mat& matK, cv::Mat& Tcw, std::vector<bool>& outlier,
                             std::vector<float>& chi2s);

    static void SlidingWindowBA(std::vector<KeyFrame*>& vKFs, const cv::Mat& matK);
//    static bool SolvePnP(Frame& frame, std::vector<MapPoint*>& vpMapPoints, cv::Mat& Tcw);
    static bool SolvePnP(const std::vector<cv::Point2f>& vPtsUn, std::vector<MapPoint*>& vpMapPoints,
                         const cv::Mat& K, cv::Mat& Tcw);
};

}

#endif //NAIVE_SLAM_OPTIMIZATION_H
