/*
 * @Author: hanfuyong
 * @Date: 2022-08-02 15:04:31
 * @LastEditors: hanfuyong
 * @LastEditTime: 2022-08-02 15:28:23
 * @FilePath: /naive_slam/src/Map.cc
 * @Description: 仅用于个人学习
 * 
 * Copyright (c) 2022 by hanfuyong, All Rights Reserved. 
 */

#include "Map.h"

namespace Naive_SLAM{

void Map::AddMapPoint(MapPoint* mapPoint){
    mvpMapPoints.emplace_back(mapPoint);
}

void Map::InsertMapPoints(const std::vector<MapPoint*>& mapPoints) {
    mvpMapPoints.insert(mvpMapPoints.end(), mapPoints.begin(), mapPoints.end());
}

MapPoint* Map::GetMapPoint(int id){
    return mvpMapPoints[id];
}

}