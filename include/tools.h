//
// Created by hanfuyong on 2022/8/17.
//

#ifndef NAIVE_SLAM_TOOLS_H
#define NAIVE_SLAM_TOOLS_H

#include <opencv2/opencv.hpp>
#include "Frame.h"
#include "MapPoint.h"

namespace Naive_SLAM{

void DrawMatches(const cv::Mat& img1, const cv::Mat& img2,
                 const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2,
                 const std::vector<cv::Point2f>& img1_points, const std::vector<cv::Point2f>& img2_points){
    int num = points1.size();
    int w = img1.size().width;
    int h = img1.size().height;
    cv::Mat imgShow(h, w * 2, CV_8UC3, cv::Scalar::all(0));
    cv::Mat tmp;
    cv::cvtColor(img1, tmp, cv::COLOR_GRAY2BGR);
    tmp.copyTo(imgShow(cv::Rect(0, 0, w, h)));
    cv::cvtColor(img2, tmp, cv::COLOR_GRAY2BGR);
    tmp.copyTo(imgShow(cv::Rect(w, 0, w, h)));

    for (size_t i = 0; i < img1_points.size(); i++) {
        cv::circle(imgShow, img1_points[i], 3, cv::Scalar(0, 0, 255));
    }
    for (size_t i = 0; i < img2_points.size(); i++) {
        cv::circle(imgShow, (img2_points[i] + cv::Point2f(w, 0)), 3, cv::Scalar(0, 0, 255));
    }
    for (size_t i = 0; i < points1.size(); i++){
        cv::circle(imgShow, points1[i], 3, cv::Scalar(0, 255, 0));
        cv::circle(imgShow, (points2[i] + cv::Point2f(w, 0)), 3, cv::Scalar(0, 255, 0));
        cv::line(imgShow, points1[i], (points2[i] + cv::Point2f(w, 0)), cv::Scalar(255, 0, 0));
    }
    cv::putText(imgShow, std::to_string(num), cv::Point(20, 20), 1, 1, cv::Scalar(255, 0, 0));
    cv::imshow("match", imgShow);
    cv::waitKey(0);
}

void DrawPoints(const cv::Mat& img, const std::vector<cv::Point2f>& pointsDet,
                const std::vector<cv::Point2f>& pointsTracked, const std::vector<bool>& bOutlier,
                const cv::Mat& mK, const cv::Mat& distCoff){
    cv::Mat imgShow, imgUn;
    cv::undistort(img, imgUn, mK, distCoff);
    cv::cvtColor(imgUn, imgShow, cv::COLOR_GRAY2BGR);

    // 画出所有检测的点
    for(int i = 0; i < pointsDet.size(); i++) {
        cv::circle(imgShow, pointsDet[i], 3, cv::Scalar(255, 255, 0));
    }

    // 画出投影匹配上的点
    int numInliers = 0;
    int n = pointsTracked.size();
    for(int i = 0; i < n; i++){
        if(!bOutlier[i]){
            cv::circle(imgShow, pointsTracked[i], 3, cv::Scalar(0, 255, 0));
            numInliers++;
        }
        else{
            cv::circle(imgShow, pointsTracked[i], 3, cv::Scalar(0, 0, 255));
        }
    }
    cv::putText(imgShow, "Total tracked points: " + std::to_string(n), cv::Point(20, 35), 1, 1, cv::Scalar(255, 0, 0));
    cv::putText(imgShow, "Inlier points: " + std::to_string(numInliers), cv::Point(20, 20), 1, 1, cv::Scalar(255, 0, 0));
    cv::imshow("Track", imgShow);
    cv::waitKey(0);
}

void DrawPoints(const Frame& frame,
                const std::vector<cv::Point2f>& pointsTracked1, const std::vector<bool>& bOutlier1,
                const std::vector<cv::Point2f>& pointsTracked2 = std::vector<cv::Point2f>(),
                const std::vector<bool>& bOutlier2 = std::vector<bool>()){
    cv::Mat imgUn, imgShow;
    cv::undistort(frame.mImg, imgUn, frame.mK, frame.mDistCoef);
    cv::cvtColor(imgUn, imgShow, cv::COLOR_GRAY2BGR);

    // 画出所有检测的点
    for(int i = 0; i < frame.mvPointsUn.size(); i++) {
        cv::circle(imgShow, frame.mvPointsUn[i], 3, cv::Scalar(255, 0, 0));
    }

    // 画出第一次优化的点
    int numInliers = 0;
    int n = pointsTracked1.size();
    for(int i = 0; i < n; i++){
        if(!bOutlier1[i]){
            cv::circle(imgShow, pointsTracked1[i], 3, cv::Scalar(0, 255, 0));
            numInliers++;
        }
        else{
            cv::circle(imgShow, pointsTracked1[i], 3, cv::Scalar(0, 0, 255));
        }
    }
    cv::putText(imgShow, "Total extracted points: " + std::to_string(frame.mvPointsUn.size()), cv::Point(20, 15), 1, 1, cv::Scalar(0, 0, 255));
    cv::putText(imgShow, "Total frist tracked points: " + std::to_string(n), cv::Point(20, 30), 1, 1, cv::Scalar(0, 0, 255));
    cv::putText(imgShow, "First inlier points: " + std::to_string(numInliers), cv::Point(20, 45), 1, 1, cv::Scalar(0, 0, 255));

    // 画出第二次投影匹配的点
    if(pointsTracked2.empty()) {
        int numInliers2 = 0;
        for (int i = 0; i < pointsTracked2.size(); i++) {
            if (!bOutlier2[i]) {
                cv::circle(imgShow, pointsTracked2[i], 3, cv::Scalar(0, 255, 255));
                numInliers2++;
            } else {
                cv::circle(imgShow, pointsTracked2[i], 3, cv::Scalar(255, 0, 255));
            }
        }
        cv::putText(imgShow, "Total second tracked points: " + std::to_string(pointsTracked2.size()), cv::Point(20, 60),
                    1, 1, cv::Scalar(0, 0, 255));
        cv::putText(imgShow, "Second inlier points: " + std::to_string(numInliers2), cv::Point(20, 75), 1, 1,
                    cv::Scalar(0, 0, 255));
    }
    cv::imshow("Track", imgShow);
    cv::waitKey(0);
}

void PrintMat(const std::string& msg, const cv::Mat& mat){
    std::cout << msg << std::endl;
    for(int i = 0; i < mat.rows; i++){
        for(int j = 0; j < mat.cols; j++){
            std::cout << mat.at<float>(i, j) << "  ";
        }
        std::cout << std::endl;
    }
}


}

#endif //NAIVE_SLAM_TOOLS_H
