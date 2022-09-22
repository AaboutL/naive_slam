/*
 * @Author: hanfuyong
 * @Date: 2022-07-27 15:10:24
 * @LastEditors: hanfuyong
 * @LastEditTime: 2022-08-02 19:06:27
 * @FilePath: /naive_slam/src/Estimator.cc
 * @Description: 仅用于个人学习
 * 
 * Copyright (c) 2022 by hanfuyong, All Rights Reserved. 
 */
#include "Estimator.h"
#include "Optimization.h"
#include "Matcher.h"
#include "tools.h"
#include <chrono>

namespace Naive_SLAM {

    Estimator::Estimator(float fx, float fy, float cx, float cy, float k1, float k2, float p1,
                         float p2) :
            fx(fx), fy(fy), cx(cx), cy(cy), mState(NOT_INITIALIZED) {
        mK = cv::Mat::eye(3, 3, CV_32FC1);
        mK.at<float>(0, 0) = fx;
        mK.at<float>(1, 1) = fy;
        mK.at<float>(0, 2) = cx;
        mK.at<float>(1, 2) = cy;
        mDistCoef = cv::Mat::zeros(4, 1, CV_32FC1);
        mDistCoef.at<float>(0, 0) = k1;
        mDistCoef.at<float>(1, 0) = k2;
        mDistCoef.at<float>(2, 0) = p1;
        mDistCoef.at<float>(3, 0) = p2;

        mpORBExtractor = new ORBextractor(300, 1.2, 1, 20, 7);
        mpORBExtractorInit = new ORBextractor(500, 1.2, 1, 20, 7);
    }

    Estimator::Estimator(const std::string &strParamFile, Map *pMap, KeyFrameDB *pKeyFrameDB,
                         Vocabulary *pORBVocabulary) :
            mState(NOT_INITIALIZED), mpKeyFrameDB(pKeyFrameDB), mpMap(pMap),
            mpORBVocabulary(pORBVocabulary) {
        cv::FileStorage fs(strParamFile.c_str(), cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cout << "[Estimator] Param file not exist..." << std::endl;
            exit(0);
        }
        fx = fs["Camera.fx"];
        fy = fs["Camera.fy"];
        cx = fs["Camera.cx"];
        cy = fs["Camera.cy"];
        mK = cv::Mat::eye(3, 3, CV_32FC1);
        mK.at<float>(0, 0) = fx;
        mK.at<float>(1, 1) = fy;
        mK.at<float>(0, 2) = cx;
        mK.at<float>(1, 2) = cy;

        mDistCoef = cv::Mat::zeros(4, 1, CV_32FC1);
        mDistCoef.at<float>(0, 0) = fs["Camera.k1"];
        mDistCoef.at<float>(1, 0) = fs["Camera.k2"];
        mDistCoef.at<float>(2, 0) = fs["Camera.p1"];
        mDistCoef.at<float>(3, 0) = fs["Camera.p2"];

        int nFeatureNum = fs["feature_num"];
        mpORBExtractorInit = new ORBextractor(nFeatureNum, fs["level_factor"], fs["pyramid_num"],
                                              fs["FAST_th_init"], fs["FAST_th_min"]);
        mpORBExtractor = new ORBextractor(nFeatureNum, fs["level_factor"], fs["pyramid_num"],
                                          fs["FAST_th_init"], fs["FAST_th_min"]);

        mImgWidth = fs["Camera.width"];
        mImgHeight = fs["Camera.height"];
        mCellSize = fs["cell_size"];
        mGridCols = (int) std::ceil((float) mImgWidth / (float) mCellSize);
        mGridRows = (int) std::ceil((float) mImgHeight / (float) mCellSize);
        mnPyramidNum = fs["pyramid_num"];
        mSlidingWindowSize = fs["SlidingWindowSize"];
        mpInitializer = new Initializer(mSlidingWindowSize, mK, mDistCoef, mpMap, mpKeyFrameDB);
    }

    void Estimator::Estimate(const cv::Mat &image, const double &timestamp) {
        mImGray = image;
        if (image.channels() == 3) {
            cv::cvtColor(mImGray, mImGray, cv::COLOR_BGR2GRAY);
        }
        if (mState == NOT_INITIALIZED) {
            mCurrentFrame = Frame(mImGray, timestamp, mpORBExtractorInit, mK, mDistCoef,
                                  mImgWidth, mImgHeight, mCellSize, mGridRows, mGridCols,
                                  mpORBVocabulary);

            auto *pKF = new KeyFrame(mCurrentFrame);
            mpInitializer->Insert(pKF, mImGray);
            if(mpInitializer->ReadyToInit()){
                mpInitializer->Initialize(mvpSlidingWindowKFs, mspSlidingWindowMPs);
                mCurrentFrame.SetT(mvpSlidingWindowKFs.back()->GetTcw());
                UpdateVelocity(mvpSlidingWindowKFs[mvpSlidingWindowKFs.size()-2]->GetTwc(),
                               mvpSlidingWindowKFs.back()->GetTcw());
                mLastFrame = Frame(mCurrentFrame);
                mLastestKFImg = mImGray.clone();
                mState = OK;
            }
            return;
        }
        if (mState == OK) {
            std::cout << "SlidingWindow KFs num: " << mvpSlidingWindowKFs.size()
                      << "    MPs num: " << mspSlidingWindowMPs.size() << std::endl;
            mCurrentFrame = Frame(mImGray, timestamp, mpORBExtractor, mK, mDistCoef,
                                  mImgWidth, mImgHeight, mCellSize, mGridRows, mGridCols,
                                  mpORBVocabulary);

            KeyFrame* pKFtmp = mvpSlidingWindowKFs.back();
            DrawPoints(mLastestKFImg, pKFtmp->GetPointsUn(), pKFtmp->GetMapPoints(), mK, mDistCoef,
                       pKFtmp->GetRotation(), pKFtmp->GetTranslation(), "LastestKFImg");
            bool bOK1 = TrackWithKeyFrame();
//            bool bOK = TrackWithOpticalFlow();
            std::cout << "[Estimate::TrackWithKeyFrame] track state: " << bOK1 << std::endl;

            bool bOK2 = TrackWithinSlidingWindow();
            std::cout << "[Estimate::TrackWithinSlidingWindow] track state: " << bOK2 << std::endl;

            if (bOK1 || bOK2) {
                if (!(bOK1 && bOK2)) {
                    KeyFrame *pNewKF = CreateKeyFrame();
                    // 新关键帧与滑窗中老的关键帧之间创建新的地图点
                    CreateNewMapPoints(pNewKF);
                    SlidingWindowBA();
                    if (mvpSlidingWindowKFs.size() < mSlidingWindowSize) {
                        mvpSlidingWindowKFs.emplace_back(pNewKF);
                        // 同时，把新关键帧对应的mappoint插入到滑窗中
                        for (auto *pMP: pNewKF->GetMapPoints()) {
                            if (pMP)
                                mspSlidingWindowMPs.insert(pMP);
                        }
                    } else {
                        // 从slidingWindow中删除旧的关键帧和地图点, 并插入新的关键帧。
                        Marginalize();
                        mvpSlidingWindowKFs.emplace_back(pNewKF);
                        // 更新地图点的滑窗，先清空，然后重新插入。
                        mspSlidingWindowMPs.clear();
                        for (auto *pKF: mvpSlidingWindowKFs) {
                            auto vpMPs = pKF->GetMapPoints();
                            for (auto *pMP: vpMPs) {
                                if (pMP)
                                    mspSlidingWindowMPs.insert(pMP);
                            }
                        }
                    }
                    mLastestKFImg = mImGray.clone();
                    std::cout << "SlidingWindow KFs num: " << mvpSlidingWindowKFs.size()
                              << "    MPs num: " << mspSlidingWindowMPs.size() << std::endl;
                }
            }

            UpdateVelocity();
        }
        mLastFrame = Frame(mCurrentFrame);
    }

/*
 * 更新运动模型
 */
    void Estimator::UpdateVelocity() {
        cv::Mat lastTwc = mLastFrame.GetTwc();
        mVelocity = mCurrentFrame.GetTcw() * lastTwc;
    }
    void Estimator::UpdateVelocity(const cv::Mat& lastTwc, const cv::Mat& currTcw){
        mVelocity = currTcw * lastTwc;
    }

    int Estimator::DescriptorDistance(const cv::Mat &a, const cv::Mat &b) {
        const int *pa = a.ptr<int32_t>();
        const int *pb = b.ptr<int32_t>();

        int dist = 0;

        for (int i = 0; i < 8; i++, pa++, pb++) {
            unsigned int v = *pa ^ *pb;
            v = v - ((v >> 1) & 0x55555555);
            v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
            dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
        }

        return dist;
    }

    std::vector<int> Estimator::SearchByProjection(const KeyFrame* pKF,
                                                   const cv::Mat &Tcw) {
        float radiusTh = 15.0;
        std::vector<MapPoint*> pMPs = pKF->GetMapPoints();
        cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
        cv::Mat tcw = Tcw.rowRange(0, 3).col(3);
        std::vector<int> matchesIdx(pKF->N, -1);
        int nMatches = 0;
        int mpNum = 0;
        for (int i = 0; i < pKF->N; i++) {
            MapPoint *pMP = pMPs[i];
            if (pMP) {
                mpNum++;
                cv::Mat pt3dw = pMP->GetWorldPos();
                cv::Mat matDesc = pMP->GetDescription();
                cv::Mat pt3dc = Rcw * pt3dw + tcw;
                cv::Point2f pt2dUn = project(pt3dc);
                if (pt2dUn.x >= mImgWidth || pt2dUn.x < 0 || pt2dUn.y < 0 || pt2dUn.y >= mImgHeight)
                    continue;

                int matchedId;
                int level = pKF->GetKeyPointUn(i).octave;
                float scaleFactor = pKF->GetScaleFactors()[level];
                float radius = radiusTh * scaleFactor;
                int minLevel = level - 1;
                int maxLevel = level + 1;
                int bestDist = Matcher::SearchGrid(pt2dUn, matDesc, mCurrentFrame,
                                        radius, matchedId, minLevel, maxLevel);
                if (bestDist < 50) {
                    matchesIdx[i] = matchedId;
                    nMatches++;
                }
            }
        }
        std::cout << "[SearchByProjection] MapPoint in Ref KF nums: " << mpNum << std::endl;
        std::cout << "[SearchByProjection] MapPoint matched nums: " << nMatches << std::endl;
        return matchesIdx;
    }

    bool Estimator::TrackWithKeyFrame() {
        KeyFrame *lastestKF = mvpSlidingWindowKFs.back();
        std::vector<MapPoint *> mapPointsInKF = lastestKF->GetMapPoints();
        cv::Mat frameTcw = mVelocity * mLastFrame.GetTcw();
        mCurrentFrame.SetT(frameTcw);

//        DrawPoints(mCurrentFrame.mImg, mCurrentFrame.mvPointsUn,
//                   mvpSlidingWindowKFs.back()->GetMapPoints(), mK, mDistCoef,
//                   frameTcw.rowRange(0, 3).colRange(0, 3), frameTcw.rowRange(0, 3).col(3),
//                   "Frame");

        std::vector<int> matchesIdx = SearchByProjection(lastestKF, frameTcw);
        std::vector<MapPoint *> mapPoints;
//        std::vector<cv::Point2f> pointsUnMatched;
        std::vector<cv::KeyPoint> vKPsUnMatched;
        for (int i = 0; i < lastestKF->N; i++) {
            if (matchesIdx[i] == -1)
                continue;
            mapPoints.emplace_back(mapPointsInKF[i]);
            vKPsUnMatched.emplace_back(mCurrentFrame.mvKeyPointsUn[matchesIdx[i]]);
        }
        std::vector<bool> bOutlier;
        std::vector<float> chi2s;
        mnKeyFrameMatchInliers = Optimization::PoseOptimize(vKPsUnMatched, mapPoints,
                                                            lastestKF->GetInvLevelSigma2(), mK,
                                                            frameTcw, bOutlier, chi2s);
        std::cout << "[TrackWithKeyFrame] g2o opt mnKeyFrameMatchInliers= "
                  << mnKeyFrameMatchInliers << std::endl;

        mCurrentFrame.SetT(frameTcw);

        // 画图
        DrawPoints(mCurrentFrame, vKPsUnMatched, bOutlier, chi2s, mapPoints);

        if (mnKeyFrameMatchInliers > 10) {
            return true;
        } else
            return false;
    }

    cv::Point2f Estimator::project(const cv::Mat &pt3d) const {
        float x_norm = pt3d.at<float>(0) / pt3d.at<float>(2);
        float y_norm = pt3d.at<float>(1) / pt3d.at<float>(2);
        float x_un = x_norm * fx + cx;
        float y_un = y_norm * fy + cy;
        return {x_un, y_un};
    }

    int Estimator::SearchByProjection(std::vector<MapPoint *> &vMapPoints,
                                      std::vector<cv::KeyPoint> &vKPsUn,
                                      std::vector<int> &vMatchedIdx) {
        vMatchedIdx.resize(mCurrentFrame.N, -1);
        cv::Mat Tcw = mCurrentFrame.GetTcw();
        cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
        cv::Mat tcw = Tcw.rowRange(0, 3).col(3);
        int nMatches = 0;
        for (const auto &pMP: mspSlidingWindowMPs) {
            cv::Mat pt3dw = pMP->GetWorldPos();
            cv::Mat matDesc = pMP->GetDescription();
            cv::Mat pt3dc = Rcw * pt3dw + tcw;
            cv::Point2f pt2dUn = project(pt3dc);
            if (pt2dUn.x >= mImgWidth || pt2dUn.x < 0 || pt2dUn.y < 0 || pt2dUn.y >= mImgHeight)
                continue;

            int matchedId;
            int bestDist = Matcher::SearchGrid(pt2dUn, matDesc, mCurrentFrame,
                                               40, matchedId, 0, mnPyramidNum);
            if (bestDist < 50) {
                vKPsUn.emplace_back(mCurrentFrame.mvKeyPointsUn[matchedId]);
                vMapPoints.emplace_back(pMP);
                vMatchedIdx[matchedId] = nMatches;
                nMatches++;
            }
        }
        return nMatches;
    }

    bool Estimator::TrackWithinSlidingWindow() {
        std::vector<MapPoint *> vMapPointsMatched;
//        std::vector<cv::Point2f> vPointsUnMatched;
        std::vector<cv::KeyPoint> vKPsUnMatched;
        std::vector<int> vMatchedIdx;
        int nMatches = SearchByProjection(vMapPointsMatched, vKPsUnMatched, vMatchedIdx);
        std::cout << "[TrackWithinSlidingWindow] project matches num: " << nMatches << std::endl;

        cv::Mat frameTcw = mCurrentFrame.GetTcw();
        std::vector<bool> bOutlier;
        std::vector<float> chi2s;
        mnSlidingWindowMatchInliers = Optimization::PoseOptimize(vKPsUnMatched,vMapPointsMatched,
                                                                 mCurrentFrame.mvInvLevelSigma2,
                                                                 mK, frameTcw, bOutlier, chi2s);
        std::cout << "[TrackWithinSlidingWindow] mnSlidingWindowMatchInliers="
                  << mnSlidingWindowMatchInliers << std::endl;

        mCurrentFrame.SetT(frameTcw);

        // 画图
        DrawPoints(mCurrentFrame, vKPsUnMatched, bOutlier, chi2s, vMapPointsMatched);
        if (mnSlidingWindowMatchInliers > 30) {
            mvpCurrentTrackedMPs.clear();
            mvpCurrentTrackedMPs.resize(mCurrentFrame.N, nullptr);
            for (int i = 0; i < mCurrentFrame.N; i++) {
                if (vMatchedIdx[i] != -1 && !bOutlier[vMatchedIdx[i]]) {
                    mvpCurrentTrackedMPs[i] = vMapPointsMatched[vMatchedIdx[i]];
                }
            }
            return true;
        } else
            return false;
    }

    bool Estimator::NeedNewKeyFrame() {
        if (mnSlidingWindowMatchInliers < 40) {
            return true;
        } else
            return false;
    }

    KeyFrame *Estimator::CreateKeyFrame() {
        auto *pCurKF = new KeyFrame(mCurrentFrame);
        pCurKF->SetMapPoints(mvpCurrentTrackedMPs);
        pCurKF->ComputeBow();
        std::cout << "[Estimator::CreateKeyFrame] done" << std::endl;
        return pCurKF;
    }

    /*
     * 新关键帧与滑窗中的所有关键帧进行BoW匹配，对匹配点创建地图点。
     */
    void Estimator::CreateNewMapPoints(KeyFrame *pKF) {
        for (auto *pKFSW: mvpSlidingWindowKFs) {
            pKFSW->ComputeBow();
            const DBoW2::FeatureVector &vFeatVec1 = pKF->GetFeatVec();
            const DBoW2::FeatureVector &vFeatVec2 = pKFSW->GetFeatVec();
            DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
            DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
            DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
            DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

//            cv::Mat F12 = ComputeF12(pKF, pKFSW);
            cv::Mat F12 = pKF->ComputeFundamental(pKFSW);

            std::vector<bool> vbMatched2(pKFSW->N, false);
            std::vector<int> vMatched12(pKF->N, -1);
            while (f1it != f1end || f2it != f2end) {
                if (f1it->first == f2it->first) {
                    for (size_t i1 = 0; i1 < f1it->second.size(); i1++) {
                        const size_t idx1 = f1it->second[i1];
                        MapPoint *pMP1 = pKF->GetMapPoint(idx1);
                        if (pMP1 && !pMP1->IsBad())
                            continue;
                        cv::KeyPoint kp1Un = pKF->GetKeyPointUn(idx1);
                        cv::Mat description1 = pKF->GetDescription(idx1);

                        int bestDist = 50;
                        int bestIdx2 = -1;
                        for (size_t i2 = 0; i2 < f2it->second.size(); i2++) {
                            const size_t idx2 = f2it->second[i2];
                            MapPoint *pMP2 = pKFSW->GetMapPoint(idx2);
                            if (vbMatched2[idx2] || (pMP2 && !pMP2->IsBad()))
                                continue;
                            cv::KeyPoint kp2Un = pKFSW->GetKeyPointUn(idx2);
                            cv::Mat description2 = pKFSW->GetDescription(idx2);
                            int dist = DescriptorDistance(description1, description2);
                            if (dist > bestDist)
                                continue;

                            if (Matcher::CheckDistEpipolarLine(kp1Un, kp2Un, F12)) {
                                bestIdx2 = idx2;
                                bestDist = dist;
                            }
                        }
                        if (bestIdx2 >= 0) {
                            vMatched12[idx1] = bestIdx2;
                            vbMatched2[bestIdx2] = true;
                        }
                    }
                    f1it++;
                    f2it++;
                } else if (f1it->first < f2it->first) {
                    f1it = vFeatVec1.lower_bound(f2it->first);
                } else {
                    f2it = vFeatVec2.lower_bound(f1it->first);
                }
            }

            // Collect Matched points
            std::vector<std::pair<int, int>> vIdxMatch12;
            std::vector<cv::Point2f> vPtsMatch1, vPtsMatch2;
            for (int i = 0; i < vMatched12.size(); i++) {
                if (vMatched12[i] == -1)
                    continue;
                vPtsMatch1.emplace_back(pKF->GetKeyPointUn(i).pt);
                vPtsMatch2.emplace_back(pKFSW->GetKeyPointUn(vMatched12[i]).pt);
                vIdxMatch12.emplace_back(std::make_pair(i, vMatched12[i]));
            }
            if(vPtsMatch1.empty())
                continue;

            cv::Mat R1w = pKF->GetRotation();
            cv::Mat t1w = pKF->GetTranslation();
            cv::Mat Rw1 = pKF->GetRotationInv();
            cv::Mat tw1 = pKF->GetCameraCenter();
            cv::Mat R2w = pKFSW->GetRotation();
            cv::Mat t2w = pKFSW->GetTranslation();
            // 计算pKF1作为currentKF，pKF2作为滑窗中的KF
            // 重建在pKF1相机位姿下的空间点，然后再转换到世界坐标系下
            // 首先要得到pKF2到pKF1的旋转平移
            cv::Mat R21 = R2w * R1w.t();
            cv::Mat t21 = -R21 * t1w + t2w;
            cv::Mat P1(3, 4, CV_32F, cv::Scalar(0));
            mK.copyTo(P1.rowRange(0, 3).colRange(0, 3));
            cv::Mat P2(3, 4, CV_32F, cv::Scalar(0));
            R21.copyTo(P2.rowRange(0, 3).colRange(0, 3));
            t21.copyTo(P2.rowRange(0, 3).col(3));
            P2 = mK * P2;
            cv::Mat points4D;
            cv::triangulatePoints(P1, P2, vPtsMatch1, vPtsMatch2, points4D);

            for (int i = 0; i < points4D.cols; i++) {
                cv::Mat p3dC1 = points4D.col(i); // pKF1相机位姿下的坐标，需要转换到世界坐标系下。
                p3dC1 = p3dC1.rowRange(0, 3) / p3dC1.at<float>(3);
                if (p3dC1.at<float>(2) <= 0) {
                    continue;
                }
                cv::Mat p3dC2 = R21 * p3dC1 + t21;
                if (p3dC2.at<float>(2) <= 0) {
                    continue;
                }

                int matchIdx1 = vIdxMatch12[i].first;
                int matchIdx2 = vIdxMatch12[i].second;
                cv::Mat description = pKF->GetDescription(matchIdx1);

                cv::Mat p3dW = Rw1 * p3dC1 + tw1;
                auto *mapPoint = new MapPoint(p3dW, pKF);
                mapPoint->SetDescription(description);
                mapPoint->AddObservation(pKF, matchIdx1);
                mapPoint->AddObservation(pKFSW, matchIdx2);
                mpMap->AddMapPoint(mapPoint);

                pKF->AddMapPoint(matchIdx1, mapPoint);
                pKFSW->AddMapPoint(matchIdx2, mapPoint);
                mspSlidingWindowMPs.insert(mapPoint);
            }
        }

    }

    void Estimator::SlidingWindowBA() {
        Optimization::SlidingWindowBA(mvpSlidingWindowKFs, mK);
    }

    void Estimator::Marginalize() {
        auto it = mvpSlidingWindowKFs.begin();
        mvpSlidingWindowKFs.erase(it);
    }

}
