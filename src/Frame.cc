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

namespace Naive_SLAM {

    float Frame::fx, Frame::fy, Frame::cx, Frame::cy;


    Frame::Frame(const Frame &frame) : N(frame.N), mTimeStamp(frame.mTimeStamp),
                                       mpORBextractor(frame.mpORBextractor),
                                       mvKeyPoints(frame.mvKeyPoints),
                                       mvKeyPointsUn(frame.mvKeyPointsUn), mvPoints(frame.mvPoints),
                                       mvPointsUn(frame.mvPointsUn),
                                       mvL0KPIndices(frame.mvL0KPIndices),
                                       mvMapPointIndices(frame.mvMapPointIndices),
                                       mDescriptions(frame.mDescriptions.clone()),
                                       mImg(frame.mImg.clone()),
                                       mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
                                       mpORBVocabulary(frame.mpORBVocabulary),
                                       mRcw(frame.mRcw.clone()), mtcw(frame.mtcw.clone()),
                                       mTcw(frame.mTcw.clone()),
                                       mRwc(frame.mRwc.clone()), mtwc(frame.mtwc.clone()),
                                       mTwc(frame.mTwc.clone()),
                                       mImgWidth(frame.mImgWidth), mImgHeight(frame.mImgHeight),
                                       mCellSize(frame.mCellSize), mGridRows(frame.mGridRows),
                                       mGridCols(frame.mGridCols),
                                       mGrid(frame.mGrid), mvScaleFactors(frame.mvScaleFactors),
                                       mvLevelSigma2(frame.mvLevelSigma2),
                                       mvInvLevelSigma2(frame.mvInvLevelSigma2){}

    Frame::Frame(const cv::Mat &img, const double &timestamp, ORBextractor *extractor,
                 const cv::Mat &K, const cv::Mat &distCoef,
                 int imgWidth, int imgHeight, int cellSize, int gridRows, int gridCols,
                 Vocabulary *pORBVocabulary) :
            mTimeStamp(timestamp), mpORBextractor(extractor), mImg(img.clone()), mK(K.clone()),
            mDistCoef(distCoef.clone()),
            mpORBVocabulary(pORBVocabulary),
            mRcw(cv::Mat::eye(3, 3, CV_32F)), mtcw(cv::Mat::zeros(3, 1, CV_32F)),
            mTcw(cv::Mat::eye(4, 4, CV_32F)),
            mRwc(cv::Mat::eye(3, 3, CV_32F)), mtwc(cv::Mat::zeros(3, 1, CV_32F)),
            mTwc(cv::Mat::eye(4, 4, CV_32F)),
            mImgWidth(imgWidth), mImgHeight(imgHeight), mCellSize(cellSize), mGridRows(gridRows),
            mGridCols(gridCols),
            mvScaleFactors(extractor->GetScaleFactors()),
            mvLevelSigma2(extractor->GetScaleSigmaSquares()),
            mvInvLevelSigma2(extractor->GetInverseScaleSigmaSquares()){
        fx = K.at<float>(0, 0);
        fy = K.at<float>(1, 1);
        cx = K.at<float>(0, 2);
        cy = K.at<float>(1, 2);

        ExtractORB(img);
        N = mvKeyPoints.size();
        std::cout << "[Frame] ORB Extracted Num=" << N << std::endl;
        UndistortKeyPoints();
        for (size_t i = 0; i < mvKeyPoints.size(); i++) {
//        cv::KeyPoint kpt = mvKeyPoints[i];
//        if (kpt.octave == 0){
            mvL0KPIndices.emplace_back(i);
            mvPoints.emplace_back(mvKeyPoints[i].pt);
            mvPointsUn.emplace_back(mvKeyPointsUn[i].pt);
//        }
        }


        mRcw = cv::Mat::eye(3, 3, CV_32F);
        mtcw = cv::Mat::zeros(3, 1, CV_32F);
        mRwc = cv::Mat::eye(3, 3, CV_32F);
        mtwc = cv::Mat::zeros(3, 1, CV_32F);


//    mGrid = std::vector<std::vector<std::vector<std::size_t>>>(mGridRows, std::vector<std::vector<std::size_t>>(mGridCols, std::vector<std::size_t>(0)));
        mGrid = new std::vector<size_t> *[mGridRows];
        for (int i = 0; i < mGridRows; i++) {
            mGrid[i] = new std::vector<size_t>[mGridCols];
        }
        AssignGrid();
    }

    void Frame::ExtractORB(const cv::Mat &img) {
        (*mpORBextractor)(img, cv::Mat(), mvKeyPoints, mDescriptions);
    }

    void Frame::UndistortKeyPoints() {
        if (mDistCoef.at<float>(0) == 0.0) {
            mvKeyPointsUn = mvKeyPoints;
            return;
        }

        cv::Mat mat(N, 2, CV_32F);
        for (int i = 0; i < N; i++) {
            mat.at<float>(i, 0) = mvKeyPoints[i].pt.x;
            mat.at<float>(i, 1) = mvKeyPoints[i].pt.y;
        }

        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
        mat = mat.reshape(1);

        mvKeyPointsUn.resize(N);
        for (int i = 0; i < N; i++) {
            cv::KeyPoint kp = mvKeyPoints[i];
            kp.pt.x = mat.at<float>(i, 0);
            kp.pt.y = mat.at<float>(i, 1);
            mvKeyPointsUn[i] = kp;
        }
    }

    /*
     * 按照去畸变之后的点坐标进行grid分配。对于未去畸变的点，不能使用此grid
     */
    void Frame::AssignGrid() {
        for (int i = 0; i < N; i++) {
            cv::Point2f ptUn = mvPointsUn[i];
            if (ptUn.x < 0 || ptUn.x >= mImgWidth || ptUn.y < 0 || ptUn.y >= mImgHeight)
                continue;
            int colIdx = int(ptUn.x / mCellSize);
            int rowIdx = int(ptUn.y / mCellSize);
            mGrid[rowIdx][colIdx].emplace_back(i);
        }
    }

//std::vector<std::vector<std::vector<std::size_t>>> Frame::GetGrid() {
//    return mGrid;
//}
    std::vector<std::size_t> **Frame::GetGrid() const {
        return mGrid;
    }

    void Frame::SetKeyPointsAndMapPointsMatchIdx(const std::vector<int> &mapPointsIdx) {
        mvMapPointIndices = mapPointsIdx;
    }

    std::vector<int> Frame::GetKeyPointsAndMapPointsMatchIdx() const {
        return mvMapPointIndices;
    }

    cv::Mat Frame::GetRotation() const {
        return mRcw;
    }

    cv::Mat Frame::GetTranslation() const {
        return mtcw;
    }

    cv::Mat Frame::GetRotationInv() const {
        return mRwc;
    }

    cv::Mat Frame::GetCameraCenter() const {
        return mtwc;
    }

    cv::Mat Frame::GetTcw() const {
        return mTcw;
    }

    cv::Mat Frame::GetTwc() const {
        return mTwc;
    }

    void Frame::SetRotation(const cv::Mat &Rcw) {
        mRcw = Rcw.clone();
        mRwc = mRcw.t();
    }

    void Frame::SetTranslation(const cv::Mat &tcw) {
        mtcw = tcw.clone();
        mtwc = -mRwc * mtcw;
    }

    void Frame::SetT(const cv::Mat &Rcw, const cv::Mat &tcw) {
        mRcw = Rcw.clone();
        mtcw = tcw.clone();
        mTcw = cv::Mat::eye(4, 4, CV_32F);
        mRcw.copyTo(mTcw.rowRange(0, 3).colRange(0, 3));
        mtcw.copyTo(mTcw.rowRange(0, 3).col(3));

        mTwc = cv::Mat::eye(4, 4, CV_32F);
        mRwc = Rcw.t();
        mtwc = -Rcw.t() * tcw;
        mRwc.copyTo(mTwc.rowRange(0, 3).colRange(0, 3));
        mtwc.copyTo(mTwc.rowRange(0, 3).col(3));
    }

    void Frame::SetT(const cv::Mat &Tcw) {
        mTcw = Tcw.clone();

        mRcw = Tcw.rowRange(0, 3).colRange(0, 3);
        mtcw = Tcw.rowRange(0, 3).col(3);

        mRwc = mRcw.t();
        mtwc = -mRwc * mtcw;

        mTwc = cv::Mat::eye(4, 4, CV_32F);
        mRwc.copyTo(mTwc.rowRange(0, 3).colRange(0, 3));
        mtwc.copyTo(mTwc.rowRange(0, 3).col(3));
    }

} // namespace Naive_SLAM
