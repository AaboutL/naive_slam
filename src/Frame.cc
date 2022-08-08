/*
 * @Author: hanfuyong
 * @Date: 2022-07-05 10:39:55
 * @LastEditors: hanfuyong
 * @LastEditTime: 2022-08-01 22:52:50
 * @FilePath: /naive_slam/src/Frame.cc
 * @Description: 仅用于个人学习
 * 
 * Copyright (c) 2022 by hanfuyong, All Rights Reserved. 
 */

#include "Frame.h"

namespace Naive_SLAM{

float Frame::fx, Frame::fy, Frame::cx, Frame::cy;


Frame::Frame(const Frame& frame): N(frame.N), mTimeStamp(frame.mTimeStamp), mpORBextractor(frame.mpORBextractor),
                                  mvKeyPoints(frame.mvKeyPoints), mvKeyPointsUn(frame.mvKeyPointsUn), mvPoints(frame.mvPoints), mvPointsUn(frame.mvPointsUn),
                                  mvL0KPIndices(frame.mvL0KPIndices), mDescriptions(frame.mDescriptions.clone()), mRcw(frame.mRcw.clone()), mtcw(frame.mtcw.clone()),
                                  mRwc(frame.mRwc.clone()), mtwc(frame.mtwc.clone()), mImg(frame.mImg.clone()), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
                                  mCellSize(frame.mCellSize), mGridRowNum(frame.mGridRowNum), mGridColNum(frame.mGridColNum), mGrid(frame.mGrid)
{}

Frame::Frame(const cv::Mat &img, const double& timestamp, ORBextractor* extractor, const cv::Mat& K, const cv::Mat& distCoef,
             const int cellSize):mTimeStamp(timestamp), mpORBextractor(extractor), mImg(img.clone()), mK(K.clone()), mDistCoef(distCoef.clone()),
                                 mImgWidth(img.cols), mImgHeight(img.rows), mCellSize(cellSize){
    fx = K.at<float>(0, 0);
    fy = K.at<float>(1, 1);
    cx = K.at<float>(0, 2);
    cy = K.at<float>(1, 2);

    ExtractORB(img);
    N = mvKeyPoints.size();
    UndistortKeyPoints();
    for (size_t i = 0; i < mvKeyPoints.size(); i++){
        cv::KeyPoint kpt = mvKeyPoints[i];
        if (kpt.octave == 0){
            mvL0KPIndices.emplace_back(i);
            mvPoints.emplace_back(mvKeyPoints[i].pt);
            mvPointsUn.emplace_back(mvKeyPointsUn[i].pt);
        }
    }


    mRcw = cv::Mat::eye(3, 3, CV_32F);
    mtcw = cv::Mat::zeros(3, 1, CV_32F);
    mRwc = cv::Mat::eye(3, 3, CV_32F);
    mtwc = cv::Mat::zeros(3, 1, CV_32F);

    mImgWidth = mImg.cols;
    mImgHeight = mImg.rows;

    mGridColNum = int(std::ceil(mImgWidth / mCellSize));
    mGridRowNum = int(std::ceil(mImgHeight / mCellSize));
    mGrid = std::vector<std::vector<std::vector<std::size_t>>>(mGridRowNum, std::vector<std::vector<std::size_t>>(mGridColNum, std::vector<std::size_t>(0)));
    AssignGrid();
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

void Frame::AssignGrid() {
    for(int i = 0; i < N; i++){
        cv::Point2f ptUn = mvPointsUn[i];
        if(ptUn.x < 0 || ptUn.x >= mImgWidth || ptUn.y < 0 || ptUn.y >= mImgHeight)
            continue;
        int colIdx = int(ptUn.x / mCellSize);
        int rowIdx = int(ptUn.y / mCellSize);
        mGrid[rowIdx][colIdx].emplace_back(i);
    }
}

std::vector<std::vector<std::vector<std::size_t>>> Frame::GetGrid() {
    return mGrid;
}

}
