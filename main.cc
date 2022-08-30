/*
 * @Author: hanfuyong
 * @Date: 2022-08-01 17:47:03
 * @LastEditors: hanfuyong
 * @LastEditTime: 2022-08-03 15:21:46
 * @FilePath: /naive_slam/main.cc
 * @Description: 仅用于个人学习
 * 
 * Copyright (c) 2022 by hanfuyong, All Rights Reserved. 
 */
#include <iostream>
//#include <glog/logging.h>

#include "VISLAM.h"

using namespace Naive_SLAM;

std::vector<std::string> ReadData(const std::string& dataPath){
    std::ifstream fin(dataPath+"data.csv", std::ios::in);
    std::string str;
    getline(fin, str);

    std::vector<std::string> vstrTimeStamps;
    
    while(!fin.eof()){
        getline(fin, str);
        std::stringstream ss(str);
        std::string strTimeStamp;
        getline(ss, strTimeStamp, ',');
        vstrTimeStamps.emplace_back(strTimeStamp);
    }
    return vstrTimeStamps;
}

int main(int argc, char** argv){
//    FLAGS_alsologtostderr = 1;
//    google::InitGoogleLogging(argv[0]);
    std::string dataPath = "/home/aal/workspace/dataset/EuRoC/mav0/cam0/";
    std::string strParamFile = "../config.yaml";
    std::string strVocabularyPath = "/home/aal/3rdparty/DBoW2/Vocabulary/ORBvoc.txt";
//    std::string strVocabularyPath = "/home/aal/3rdparty/DBoW2/Vocabulary/ORBvoc.bin";
    VISLAM vislam(strParamFile, strVocabularyPath);
    std::vector<std::string> vstrTimeStamp = ReadData(dataPath);

    std::string strTimeStamp;
    for (int i = 0; i < vstrTimeStamp.size(); i++){
        strTimeStamp = vstrTimeStamp[i];
        double timestamp = std::stod(strTimeStamp) / 1e9;
        std::string path = dataPath + "data/" + strTimeStamp + ".png";
        std::cout << std::endl << std::endl;
        std::cout << path << "    Img id: " << i << std::endl;
        cv::Mat image = cv::imread(path);
        vislam.Run(image, timestamp);
    }
    return 0;
}