//
// Created by hanfuyong on 2022/8/17.
//

#ifndef NAIVE_SLAM_TOOLS_H
#define NAIVE_SLAM_TOOLS_H

#include <opencv2/opencv.hpp>

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

}

#endif //NAIVE_SLAM_TOOLS_H
