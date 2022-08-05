/*
 * @Author: hanfuyong
 * @Date: 2022-07-27 15:19:53
 * @LastEditors: hanfuyong
 * @LastEditTime: 2022-07-30 01:09:38
 * @FilePath: /naive_slam/src/KeyFrame.cc
 * @Description: 仅用于个人学习
 * 
 * Copyright (c) 2022 by hanfuyong, All Rights Reserved. 
 */
#include "KeyFrame.h"

namespace Naive_SLAM{
KeyFrame::KeyFrame(const Frame& frame){
    mvKeyPoints = frame.mvKeyPoints;
    mvKeyPointsUn = frame.mvKeyPointsUn;
    mDescriptions = frame.mDescriptions;
    mRcw = frame.mRcw;
    mtcw = frame.mtcw;
    mRwc = frame.mRwc;
    mtwc = frame.mtwc;
}

void KeyFrame::AddMapPoints(std::vector<MapPoint*>& mapPoints){
    mvpMapPoints = mapPoints;
}

}