/*
 * @Author: hanfuyong
 * @Date: 2022-08-02 14:54:12
 * @LastEditors: hanfuyong
 * @LastEditTime: 2022-08-02 18:46:43
 * @FilePath: /naive_slam/include/Map.h
 * @Description: 仅用于个人学习
 * 
 * Copyright (c) 2022 by hanfuyong, All Rights Reserved. 
 */
#pragma once
#include <iostream>

#include "MapPoint.h"

namespace Naive_SLAM{

class Map{
public:
    // Map();
    void AddMapPoint(MapPoint* mapPoint);
    void InsertMapPoints(const std::vector<MapPoint*>& mapPoints);
    MapPoint* GetMapPoint(int id);

private:
    std::vector<MapPoint*> mvpMapPoints;
};

}