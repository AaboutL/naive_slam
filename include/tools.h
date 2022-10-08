//
// Created by hanfuyong on 2022/8/17.
//

#ifndef NAIVE_SLAM_TOOLS_H
#define NAIVE_SLAM_TOOLS_H

#include <chrono>
#include <opencv2/opencv.hpp>
#include "Frame.h"
#include "MapPoint.h"
#include "KeyFrame.h"

namespace Naive_SLAM{

cv::Point2f projectPoint(const cv::Mat &pt3d, const cv::Mat& mK);

void DrawMatches(const std::string& winName, const cv::Mat& img1, const cv::Mat& img2,
                 const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2,
                 const std::vector<cv::Point2f>& img1_points, const std::vector<cv::Point2f>& img2_points,
                 const cv::Mat& mK, const cv::Mat& distCoff);

void DrawPoints(const cv::Mat& img, const std::vector<cv::Point2f>& pointsDet,
                const std::vector<cv::Point2f>& pointsTracked, const std::vector<bool>& bOutlier,
                const cv::Mat& mK, const cv::Mat& distCoff);

void DrawPoints(const Frame& frame, const std::vector<MapPoint*>& vpMapPoints);

void DrawPoints(const cv::Mat& img, const std::vector<cv::Point2f>& vPts,
                const std::vector<MapPoint*>& vMPs,
                const cv::Mat& mK, const cv::Mat& distCoff,
                const cv::Mat& Rcw, const cv::Mat& tcw, const std::string& winName, int s=1);

void DrawPoints(const cv::Mat& img, const KeyFrame* pKF,
                const cv::Mat& mK, const cv::Mat& distCoff,
                const std::string& winName, int s=1, std::vector<float> chi2 = std::vector<float>());

void PrintMat(const std::string& msg, const cv::Mat& mat);

void PrintTime(const std::string& msg, const std::chrono::system_clock::time_point& t);

}

#endif //NAIVE_SLAM_TOOLS_H
