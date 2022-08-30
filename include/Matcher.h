//
// Created by hanfuyong on 2022/8/30.
//

#ifndef NAIVE_SLAM_MATCHER_H
#define NAIVE_SLAM_MATCHER_H

#include "KeyFrame.h"

namespace Naive_SLAM{
class Matcher{
public:
    static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

    static int SearchByBow(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<int>& matches);
};
}

#endif //NAIVE_SLAM_MATCHER_H
