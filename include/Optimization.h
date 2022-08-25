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
    static int PoseOptimize(const std::vector<cv::Point2f>& ptsUn, const std::vector<MapPoint*>& mapPoints,
                             const cv::Mat& matK, cv::Mat& Tcw, std::vector<bool>& outlier);

    static void SlidingWindowBA(std::vector<KeyFrame*>& vKFs, const cv::Mat& matK);
};

}

#endif //NAIVE_SLAM_OPTIMIZATION_H
