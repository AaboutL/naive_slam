/*
 * @Author: hanfuyong
 * @Date: 2022-08-02 16:17:32
 * @LastEditors: hanfuyong
 * @LastEditTime: 2022-08-02 18:54:08
 * @FilePath: /naive_slam/src/VISLAM.cc
 * @Description: 仅用于个人学习
 * 
 * Copyright (c) 2022 by hanfuyong, All Rights Reserved. 
 */

#include "VISLAM.h"

namespace Naive_SLAM{

VISLAM::VISLAM(std::string& paramFilePath){
    mpMap = new Map();
    mpKeyFrameDB = new KeyFrameDB();
    mpEstimator = new Estimator(paramFilePath, mpMap, mpKeyFrameDB);
}

void VISLAM::Run(const cv::Mat& image, const double& timestamp){
    mpEstimator->Estimate(image, timestamp);
}


}