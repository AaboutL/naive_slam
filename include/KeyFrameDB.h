/*
 * @Author: hanfuyong
 * @Date: 2022-08-02 15:33:40
 * @LastEditors: hanfuyong
 * @LastEditTime: 2022-08-02 18:46:03
 * @FilePath: /naive_slam/include/KeyFrameDB.h
 * @Description: 仅用于个人学习
 * 
 * Copyright (c) 2022 by hanfuyong, All Rights Reserved. 
 */

#pragma once
#include "KeyFrame.h"

namespace Naive_SLAM{

class KeyFrameDB{
public:
    // KeyFrameDB();
    void AddKeyFrame(KeyFrame* keyFrame){
        mspKeyFrames.insert(keyFrame);
    }

private:
    std::set<KeyFrame*> mspKeyFrames;
};

}