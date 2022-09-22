//
// Created by hanfuyong on 2022/8/30.
//

#ifndef NAIVE_SLAM_MATCHER_H
#define NAIVE_SLAM_MATCHER_H

#include "KeyFrame.h"
#include "Frame.h"

namespace Naive_SLAM{
class Matcher{
public:
    static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

    static int SearchByBow(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<int>& matches);

    static int CollectMatches(const KeyFrame* pKF1, const KeyFrame* pKF2,
                                 std::vector<int>& vMatches, std::vector<int>& vMatchIdx,
                                 std::vector<cv::Point2f>& vPts1, std::vector<cv::Point2f>& vPts2,
                                 float& average_parallax,
                                 std::vector<cv::Point2f>& vPts1_tmp,
                                 std::vector<cv::Point2f>& vPts2_tmp);

    static int CollectMatches(const KeyFrame* pKF1, const KeyFrame* pKF2, const cv::Mat& F12,
                              std::vector<int>& vMatches, std::vector<int>& vMatchIdx,
                              std::vector<cv::Point2f>& vPts1, std::vector<cv::Point2f>& vPts2);

    static int SearchGrid(const cv::Point2f &pt2d, const cv::Mat &description, KeyFrame* pKF2,
                           float radius, int &matchedId, int minLevel=0, int maxLevel=0);
    static int SearchGrid(const cv::Point2f &pt2d, const cv::Mat &description, Frame& frame,
                           float radius, int &matchedId, int minLevel=0, int maxLevel=0);
    static std::vector<int> SearchByOpticalFlow(KeyFrame* pKF1, KeyFrame* pKF2,
                                         const cv::Mat& img1, const cv::Mat& img2);

    static bool CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2,
                                      const cv::Mat &F12);
};
}

#endif //NAIVE_SLAM_MATCHER_H
