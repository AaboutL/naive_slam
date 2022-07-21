/*
 * @Author: hanfuyong
 * @Date: 2022-07-05 10:39:55
 * @LastEditors: hanfuyong
 * @LastEditTime: 2022-07-06 12:18:44
 * @FilePath: /naive_slam/src/Frame.cc
 * @Description: 仅用于个人学习
 * 
 * Copyright (c) 2022 by hanfuyong, All Rights Reserved. 
 */

#include "Frame.h"

namespace Naive_SLAM{

float Frame::fx, Frame::fy, Frame::cx, Frame::cy;

Frame::Frame(const cv::Mat &img, const double& timestamp, ORBextractor* extractor, const cv::Mat& K, const cv::Mat& distCoef):
mTimeStamp(timestamp), mpORBextractor(extractor), mK(K), mDistCoef(distCoef){
    fx = K.at<float>(0, 0);
    fy = K.at<float>(1, 1);
    cx = K.at<float>(0, 2);
    cy = K.at<float>(1, 2);

    ExtractORB(img);
    UndistortKeyPoints();
}

void Frame::ExtractORB(const cv::Mat& img){
    (*mpORBextractor)(img, cv::Mat(), mvKeyPoints, mDescriptions);
}

void Frame::UndistortKeyPoints(){
    if (mDistCoef.at<float>(0) == 0.0){
        mvKeyPointsUn = mvKeyPoints;
        return;
    }

    cv::Mat mat(N, 2, CV_32F);
    for (int i = 0; i < N; i++){
        mat.at<float>(i, 0) = mvKeyPoints[i].pt.x;
        mat.at<float>(i, 1) = mvKeyPoints[i].pt.y;
    }
    
    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
    mat = mat.reshape(1);
    
    mvKeyPointsUn.resize(N);
    for (int i = 0; i < N; i++){
        cv::KeyPoint kp = mvKeyPoints[i];
        kp.pt.x = mat.at<float>(i, 0);
        kp.pt.y = mat.at<float>(i, 1);
        mvKeyPointsUn[i] = kp;
    }
}

}
