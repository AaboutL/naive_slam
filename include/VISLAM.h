/*
 * @Author: hanfuyong
 * @Date: 2022-08-02 16:10:33
 * @LastEditors: hanfuyong
 * @LastEditTime: 2022-08-02 18:23:58
 * @FilePath: /naive_slam/include/VISLAM.h
 * @Description: 仅用于个人学习
 * 
 * Copyright (c) 2022 by hanfuyong, All Rights Reserved. 
 */

#ifndef NAIVE_SLAM_VISLAM_H
#define NAIVE_SLAM_VISLAM_H
#include <iostream>

#include "Map.h"
#include "KeyFrameDB.h"
#include "Estimator.h"
#include "Vocabulary.h"

namespace Naive_SLAM{
class VISLAM{
public:
    VISLAM(std::string& paramFilePath, std::string& vocabularyPath);
    void Run(const cv::Mat& image, const double& timestamp);
private:
    Vocabulary *mpORBvocabulary;
    Map* mpMap;
    KeyFrameDB* mpKeyFrameDB;
    Estimator* mpEstimator;

};
}
#endif