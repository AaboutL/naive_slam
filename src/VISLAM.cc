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

VISLAM::VISLAM(std::string& paramFilePath,
               std::string& vocabularyPath){
    mpORBvocabulary = new Vocabulary();

    std::cout << "[VISLAM] loading vocabulary..." << std::endl;
    mpORBvocabulary->loadFromTextFile(vocabularyPath);
//    mpORBvocabulary->loadFromBinaryFile(vocabularyPath);
    std::cout << "[VISLAM] vocabulary loaded" << std::endl;

    mpMap = new Map();
    mpKeyFrameDB = new KeyFrameDB();
    mpEstimator = new Estimator(paramFilePath, mpMap, mpKeyFrameDB, mpORBvocabulary);
}

void VISLAM::Run(const cv::Mat& image, const double& timestamp){
    mpEstimator->Estimate(image, timestamp);
}


}