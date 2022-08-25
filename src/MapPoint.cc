/*
 * @Author: hanfuyong
 * @Date: 2022-07-30 00:44:25
 * @LastEditors: hanfuyong
 * @LastEditTime: 2022-08-02 14:42:08
 * @FilePath: /naive_slam/src/MapPoint.cc
 * @Description: 仅用于个人学习
 * 
 * Copyright (c) 2022 by hanfuyong, All Rights Reserved. 
 */
#include "MapPoint.h"

namespace Naive_SLAM{

MapPoint::MapPoint(const MapPoint& mapPoint){
    mmObservations = mapPoint.mmObservations;
    mpRefKF = mapPoint.mpRefKF;
    mapPoint.mWorldPos.copyTo(mWorldPos);
    mapPoint.mDescription.copyTo(mDescription);
}

MapPoint::MapPoint(const cv::Mat& mp, KeyFrame* pRefKF){
    mWorldPos = mp.clone();
    mpRefKF = pRefKF;
}

void MapPoint::AddKeyFrame(KeyFrame* pKF){
    mvpKFs.emplace_back(pKF);
}

cv::Mat MapPoint::GetWorldPos() const {
    return mWorldPos;
}

void MapPoint::SetWorldPos(const cv::Mat &worldPos) {
    mWorldPos = worldPos;
}

cv::Mat MapPoint::GetDescription() const {
    return mDescription;
}

void MapPoint::AddObservation(KeyFrame *pKF, int id) {
    mmObservations[pKF] = id;

}

void MapPoint::EraseObservation(KeyFrame *pKF) {
    mmObservations.erase(pKF);
    if(pKF == mpRefKF){
        mpRefKF = mmObservations.begin()->first;
    }
}

int MapPoint::GetIdxInKF(KeyFrame *pKF) {
    if(mmObservations.count(pKF))
        return mmObservations[pKF];
    else
        return -1;
}

int MapPoint::GetObsNum() const {
    return static_cast<int>(mmObservations.size());
}


}