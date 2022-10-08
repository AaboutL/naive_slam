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
            mpORBVocabulary(pORBVocabulary), mbRelocalized(false) {
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

        mCurrentFrame = Frame(mImGray, timestamp, mpORBExtractor, mK, mDistCoef,
                              mImgWidth, mImgHeight, mCellSize, mGridRows, mGridCols,
                              mpORBVocabulary);
        bool bOK;
        if (mState == OK) {
            std::cout << "SlidingWindow KFs num: " << mvpSlidingWindowKFs.size()
                      << "    MPs num: " << mspSlidingWindowMPs.size() << std::endl;

            KeyFrame *pKFtmp = mvpSlidingWindowKFs.back();
            DrawPoints(mLastestKFImg, pKFtmp->GetPointsUn(), pKFtmp->GetMapPoints(), mK, mDistCoef,
                       pKFtmp->GetRotation(), pKFtmp->GetTranslation(), "LastestKFImg");
            bOK = TrackWithKeyFrame();
            std::cout << "[Estimate::TrackWithKeyFrame] track state: " << bOK << std::endl;

            if (bOK) {
                bOK = TrackWithinSlidingWindow();
            }
        }
        else{
            bOK = Relocalization();
        }

        if(bOK)
            mState = OK;
        else
            mState = LOST;

        if(bOK) {
            if (NeedNewKeyFrame()) {
                KeyFrame *pNewKF = CreateKeyFrame();
                // 新关键帧与滑窗中老的关键帧之间创建新的地图点
                CreateNewMapPoints(pNewKF);
                mvpSlidingWindowKFs.emplace_back(pNewKF);
                SlidingWindowBA();
                if (mvpSlidingWindowKFs.size() < mSlidingWindowSize) {
//                    mvpSlidingWindowKFs.emplace_back(pNewKF);
                    // 同时，把新关键帧对应的mappoint插入到滑窗中
                    for (auto *pMP: pNewKF->GetMapPoints()) {
                        if (pMP)
                            mspSlidingWindowMPs.insert(pMP);
                    }
                } else {
                    // 从slidingWindow中删除旧的关键帧和地图点, 并插入新的关键帧。
                    Marginalize();
//                    mvpSlidingWindowKFs.emplace_back(pNewKF);
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

    bool Estimator::TrackWithKeyFrame() {
        KeyFrame *lastestKF = mvpSlidingWindowKFs.back();
        mvpCurrentTrackedMPs.clear();
        mvpCurrentTrackedMPs.resize(mCurrentFrame.N, nullptr);
        cv::Mat frameTcw = mVelocity * mLastFrame.GetTcw();
        mCurrentFrame.SetT(frameTcw);

        float radiusTh = 15;
        std::vector<int> vMatches;
        int nMatchesNum = Matcher::SearchByProjection(mCurrentFrame, lastestKF,
                                                      mvpCurrentTrackedMPs, frameTcw, radiusTh);
        if(nMatchesNum < 20){
            nMatchesNum = Matcher::SearchByProjection(mCurrentFrame, lastestKF,
                                                      mvpCurrentTrackedMPs, frameTcw, radiusTh*2);
        }
        if(nMatchesNum >= 20){
//            std::vector<MapPoint *> mapPoints;
//            std::vector<cv::KeyPoint> vKPsUnMatched;
//            for (int i = 0; i < lastestKF->N; i++) {
//                if (vMatches[i] == -1)
//                    continue;
//                mapPoints.emplace_back(mapPointsInKF[i]);
//                vKPsUnMatched.emplace_back(mCurrentFrame.mvKeyPointsUn[vMatches[i]]);
//            }
            std::vector<bool> bOutlier;
            std::vector<float> chi2s;
            mnKeyFrameMatchInliers = Optimization::PoseOptimize(mCurrentFrame.mvKeyPointsUn,
                                                                mvpCurrentTrackedMPs,
                                                                lastestKF->GetInvLevelSigma2(), mK,
                                                                frameTcw, bOutlier, chi2s);
            std::cout << "[Estimator::TrackWithKeyFrame] g2o opt mnKeyFrameMatchInliers= "
                      << mnKeyFrameMatchInliers << std::endl;

            if (mnKeyFrameMatchInliers >= 10) {
                mCurrentFrame.SetT(frameTcw);
                std::cout << "[Estimator::TrackWithKeyFrame] By Projection good" << std::endl;
                DrawPoints(mCurrentFrame, mvpCurrentTrackedMPs);
                return true;
            }
        }

        if (nMatchesNum < 20 || mnKeyFrameMatchInliers < 10){
            vMatches.clear();
            nMatchesNum = Matcher::SearchByBow(lastestKF, mCurrentFrame, mvpCurrentTrackedMPs);
            if(nMatchesNum < 15)
                return false;
//            std::vector<MapPoint *> mapPoints;
//            std::vector<cv::KeyPoint> vKPsUnMatched;
//            for (int i = 0; i < lastestKF->N; i++) {
//                if (vMatches[i] == -1)
//                    continue;
//                mapPoints.emplace_back(mapPointsInKF[i]);
//                vKPsUnMatched.emplace_back(mCurrentFrame.mvKeyPointsUn[vMatches[i]]);
//            }
            std::vector<bool> bOutlier;
            std::vector<float> chi2s;
            mnKeyFrameMatchInliers = Optimization::PoseOptimize(mCurrentFrame.mvKeyPointsUn,
                                                                mvpCurrentTrackedMPs,
                                                                lastestKF->GetInvLevelSigma2(), mK,
                                                                frameTcw, bOutlier, chi2s);
            std::cout << "[Estimator::TrackWithKeyFrame] g2o opt mnKeyFrameMatchInliers= "
                      << mnKeyFrameMatchInliers << std::endl;
            if(mnKeyFrameMatchInliers >=10){
                std::cout << "[Estimator::TrackWithKeyFrame] By Bow good" << std::endl;
                DrawPoints(mCurrentFrame, mvpCurrentTrackedMPs);
                mCurrentFrame.SetT(frameTcw);
                return true;
            }
            else{
                return false;
            }
        }
        return false;
    }

    cv::Point2f Estimator::project(const cv::Mat &pt3d) const {
        float x_norm = pt3d.at<float>(0) / pt3d.at<float>(2);
        float y_norm = pt3d.at<float>(1) / pt3d.at<float>(2);
        float x_un = x_norm * fx + cx;
        float y_un = y_norm * fy + cy;
        return {x_un, y_un};
    }

    bool Estimator::TrackWithinSlidingWindow() {
//        std::vector<MapPoint *> vMapPointsMatched;
//        std::vector<cv::KeyPoint> vKPsUnMatched;
//        std::vector<int> vMatchedIdx;
        int nMatches = Matcher::SearchByProjection(mCurrentFrame, mspSlidingWindowMPs,
                                                   mvpCurrentTrackedMPs);
//                                                   vMapPointsMatched, vKPsUnMatched, vMatchedIdx);
        std::cout << "[TrackWithinSlidingWindow] project matches num: " << nMatches << std::endl;

        cv::Mat frameTcw = mCurrentFrame.GetTcw();
        std::vector<bool> bOutlier;
        std::vector<float> chi2s;
        mnSlidingWindowMatchInliers = Optimization::PoseOptimize(mCurrentFrame.mvKeyPointsUn,
                                                                 mvpCurrentTrackedMPs,
                                                                 mCurrentFrame.mvInvLevelSigma2,
                                                                 mK, frameTcw, bOutlier, chi2s);
        std::cout << "[TrackWithinSlidingWindow] mnSlidingWindowMatchInliers="
                  << mnSlidingWindowMatchInliers << std::endl;

        mCurrentFrame.SetT(frameTcw);

        // 画图
        DrawPoints(mCurrentFrame, mvpCurrentTrackedMPs);
        if (mnSlidingWindowMatchInliers > 30) {
//            mvpCurrentTrackedMPs.clear();
//            mvpCurrentTrackedMPs.resize(mCurrentFrame.N, nullptr);
//            for (int i = 0; i < mCurrentFrame.N; i++) {
//                if (vMatchedIdx[i] != -1 && !bOutlier[vMatchedIdx[i]]) {
//                    mvpCurrentTrackedMPs[i] = vMapPointsMatched[vMatchedIdx[i]];
//                }
//            }
            std::cout << "[Estimator::TrackWithinSlidingWindow] good" << std::endl;
            return true;
        } else{
            std::cout << "[Estimator::TrackWithinSlidingWindow] bad" << std::endl;
            return false;
        }
    }

    bool Estimator::NeedNewKeyFrame() {
        std::cout << "[Estimator::NeedNewKeyFrame] ";
        int nMPNumInLastestKF = mvpSlidingWindowKFs.back()->GetMapPointNum();
        if (mnKeyFrameMatchInliers < 20 || mnSlidingWindowMatchInliers < 0.5*nMPNumInLastestKF
            || mbRelocalized) {
            mbRelocalized = false;
            std::cout << "Yes" << std::endl;
            return true;
        } else {
            std::cout << "No" << std::endl;
            return false;
        }
    }

    KeyFrame *Estimator::CreateKeyFrame() {
        std::cout << "[Estimator::CreateKeyFrame] start" << std::endl;
        auto *pCurKF = new KeyFrame(mCurrentFrame);
        pCurKF->SetMapPoints(mvpCurrentTrackedMPs);
        // 设置关键帧和地图点之间的观测关系
        int nTrackedMPNum = 0;
        for (int i = 0; i < pCurKF->N; i++){
            if(mvpCurrentTrackedMPs[i]) {
                mvpCurrentTrackedMPs[i]->AddObservation(pCurKF, i);
                nTrackedMPNum++;
            }
        }
        pCurKF->ComputeBow();
        std::cout << "[Estimator::CreateKeyFrame] Tracked MP in new KF num=" << nTrackedMPNum << std::endl;
        return pCurKF;
    }

    /*
     * 新关键帧与滑窗中的所有关键帧进行BoW匹配，对匹配点创建地图点。
     */
    void Estimator::CreateNewMapPoints(KeyFrame *pKF) {
        std::cout << "[Estimator::CreateNewMapPoints] start" << std::endl;
        int nNewMPNum = 0;
        for (auto *pKFSW: mvpSlidingWindowKFs) {
            pKFSW->ComputeBow();
            const DBoW2::FeatureVector &vFeatVec1 = pKF->GetFeatVec();
            const DBoW2::FeatureVector &vFeatVec2 = pKFSW->GetFeatVec();
            DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
            DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
            DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
            DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

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

//                            if (Matcher::CheckDistEpipolarLine(kp1Un, kp2Un, F12)) {
                                bestIdx2 = idx2;
                                bestDist = dist;
//                            }
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
            std::cout << "[Estimator::CreateNewMapPoints] BoW matched num=" << vPtsMatch1.size() << std::endl;

            for (int i = 0; i < points4D.cols; i++) {
                int matchIdx1 = vIdxMatch12[i].first;
                int matchIdx2 = vIdxMatch12[i].second;

                cv::Mat p3dC1 = points4D.col(i); // pKF1相机位姿下的坐标，需要转换到世界坐标系下。
                p3dC1 = p3dC1.rowRange(0, 3) / p3dC1.at<float>(3);
                if(!CheckPt3DValid(p3dC1, pKF->GetKeyPointUn(matchIdx1).pt))
                    continue;

                cv::Mat p3dC2 = R21 * p3dC1 + t21;
                if(!CheckPt3DValid(p3dC2, pKFSW->GetKeyPointUn(matchIdx2).pt))
                    continue;

                cv::Mat description = pKF->GetDescription(matchIdx1);

                cv::Mat p3dW = Rw1 * p3dC1 + tw1;
                auto *pMPNew = new MapPoint(p3dW, pKF);
                pMPNew->SetDescription(description);
                pMPNew->AddObservation(pKF, matchIdx1);
                pMPNew->AddObservation(pKFSW, matchIdx2);
                mpMap->AddMapPoint(pMPNew);

                pKF->AddMapPoint(matchIdx1, pMPNew);
                pKFSW->AddMapPoint(matchIdx2, pMPNew);
                mspSlidingWindowMPs.insert(pMPNew);
                nNewMPNum++;
            }
        }
        std::cout << "[Estimator::CreateNewMapPoints] New MP num=" << nNewMPNum << std::endl;
        std::cout << "[Estimator::CreateNewMapPoints] done" << std::endl;
    }

    void Estimator::SlidingWindowBA() {
        Optimization::SlidingWindowBA(mvpSlidingWindowKFs, mK);
    }

    void Estimator::Marginalize() {
        auto it = mvpSlidingWindowKFs.begin();
        mvpSlidingWindowKFs.erase(it);
    }

    bool Estimator::Relocalization() {
        std::cout << "[Estimator::Relocalization] start" << std::endl;
        mCurrentFrame.ComputeBow();
        for(int i = mSlidingWindowSize - 2; i >= 0; i--){
            KeyFrame* pKF = mvpSlidingWindowKFs[i];
            std::vector<int> vMatches;
            std::vector<MapPoint*> vpMapPoints(mCurrentFrame.N, nullptr);
            int nMatchesNumBow = Matcher::SearchByBow(pKF, mCurrentFrame, vpMapPoints);
            if(nMatchesNumBow < 15)
                continue;
//            std::vector<MapPoint *> mapPoints;
//            std::vector<cv::KeyPoint> vKPsUnMatched;
//            std::vector<cv::Point3f> vPts3D;
//            std::vector<cv::Point2f> vPts2D;
//            std::vector<int> vMatchIdx;
//            for (int id = 0; id < mCurrentFrame.N; id++) {
//                MapPoint* pMP = vpMapPoints[id];
//                if (!pMP || pMP->IsBad())
//                    continue;
//                vMatchIdx.emplace_back(id);
//                vPts3D.emplace_back(pMP->GetWorldPos());
//                vPts2D.emplace_back(mCurrentFrame.mvKeyPointsUn[id].pt);
//            }
//            cv::Mat rcw, tcw, Rcw, Tcw, inliers;
//            cv::solvePnPRansac(vPts3D, vPts2D, mK, cv::Mat::zeros(4, 1, CV_32F),
//                               rcw, tcw, false, 100, 4, 0.99, inliers, cv::SOLVEPNP_EPNP);
//            std::cout << "[Estimator::Relocalization] pnp inliers num=" << inliers.total() << std::endl;
//            if(inliers.total() < 10)
//                continue;
//            cv::Rodrigues(rcw, Rcw);
//            Rcw.convertTo(Rcw, CV_32F);
//            tcw.convertTo(tcw, CV_32F);
//            Tcw = cv::Mat::eye(4, 4, CV_32F);
//            Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
//            tcw.copyTo(Tcw.rowRange(0, 3).col(3));
//            std::vector<int> vMatchIdx2, vMatches2(pKF->N, -1);
//            for(int k = 0; k < inliers.total(); k++){
//                mapPoints.emplace_back(pKF->GetMapPoint(vMatchIdx[inliers.at<int>(k)]));
//                vKPsUnMatched.emplace_back(mCurrentFrame.mvKeyPointsUn[vMatches[vMatchIdx[inliers.at<int>(k)]]]);
//                vMatchIdx2.emplace_back(vMatchIdx[inliers.at<int>(k)]);
//                vMatches2[vMatchIdx[inliers.at<int>(k)]] = vMatches[vMatchIdx[inliers.at<int>(k)]];
//            }
            cv::Mat Tcw;
            bool bPnP = Optimization::SolvePnP(mCurrentFrame.mvPointsUn, vpMapPoints, mK, Tcw);
            if(!bPnP)
                continue;

            std::vector<bool> bOutlier;
            std::vector<float> chi2s;
            int nGood = Optimization::PoseOptimize(mCurrentFrame.mvKeyPointsUn, vpMapPoints,
                                                   pKF->GetInvLevelSigma2(), mK,
                                                   Tcw, bOutlier, chi2s);
            if(nGood < 10)
                continue;

            if(nGood < 50){
                int nMatchesNumProj = Matcher::SearchByProjection(mCurrentFrame, pKF,
                                                                  vpMapPoints, Tcw, 15);
                if(nGood + nMatchesNumProj >= 50){
                    nGood = Optimization::PoseOptimize(mCurrentFrame.mvKeyPointsUn, vpMapPoints,
                                                           pKF->GetInvLevelSigma2(), mK,
                                                           Tcw, bOutlier, chi2s);
                }
            }
            if(nGood >= 50){
                mCurrentFrame.SetT(Tcw);
                mbRelocalized = true;
                std::cout << "[Estimator::Relocalization] done good" << std::endl;
                return true;
            }
        }
        std::cout << "[Estimator::Relocalization] done bad" << std::endl;
        return false;
    }

    void Estimator::Reset() {

    }

    bool Estimator::CheckPt3DValid(const cv::Mat& pt3D, const cv::Point2f& ptUn) {
        if(pt3D.at<float>(2) <= 0)
            return false;
        cv::Point2f ptProj = project(pt3D);
        float dist = sqrt((ptProj.x - ptUn.x) * (ptProj.x - ptUn.x) +
                          (ptProj.y - ptUn.y) * (ptProj.y - ptUn.y));
        if (dist > 2)
            return false;
        return true;
    }

}
