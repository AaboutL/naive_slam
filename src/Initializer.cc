//
// Created by hanfuyong on 2022/8/31.
//

#include "Initializer.h"
#include "Optimization.h"
#include "tools.h"

namespace Naive_SLAM{

Initializer::Initializer(int nSlidingWindowSize, const cv::Mat& K, const cv::Mat& distCoef,
                         Map* pMap, KeyFrameDB* pKeyFrameDB):
                         mnSlidingWindowSize(nSlidingWindowSize),
                         mK(K.clone()), mDistCoef(distCoef.clone()), mpInitKF(nullptr),
                         mnStartId(0), mpMap(pMap), mpKeyFrameDB(pKeyFrameDB){
}

void Initializer::Insert(KeyFrame* pKF, cv::Mat& img){
    mKFImgs.emplace_back(img);
    if(mvpSlidingWindowKFs.empty()){
        mpInitKF = pKF;
        mvpSlidingWindowKFs.emplace_back(pKF);
        mStartImg = img.clone();
    }
    else{
        KeyFrame* pStartKF = mvpSlidingWindowKFs[mnStartId];
        mvpSlidingWindowKFs.emplace_back(pKF);
        if(mnStartId == mvpSlidingWindowKFs.size())
            return;

        if(mmMatchPairs.find(mnStartId) == mmMatchPairs.end()){
            mmMatchPairs[mnStartId] = std::vector<LinkInfo>();
        }
        std::vector<int> vMatches;
        std::chrono::system_clock::time_point t1 = std::chrono::system_clock::now();
//        int matchNum = Matcher::SearchByBow(pStartKF, pKF, vMatches);
        vMatches = Matcher::SearchByOpticalFlow(pStartKF, pKF, mStartImg, img);

        std::vector<cv::Point2f> vPtsUn0, vPtsUni, vPts1Tmp, vPts2Tmp;
        std::vector<int> vMatchIdx(pStartKF->N, -1); // 关键点与三角化点的下标的对应关系
        float average_parallax;
        int matchNum = Matcher::CollectMatches(pStartKF, pKF, vMatches, vMatchIdx, vPtsUn0, vPtsUni,
                                               average_parallax, vPts1Tmp, vPts2Tmp);
//        DrawMatches("UndistortMatch", mStartImg, img, vPtsUn0, vPtsUni, vPtsUn0, vPtsUni, mK, mDistCoef);
        DrawMatches("NormalMatch", mStartImg, img, vPts1Tmp, vPts2Tmp,
                    pStartKF->GetPointsLevel0(), pKF->GetPointsLevel0(), cv::Mat(), cv::Mat());

        std::cout << "[Initializer::Insert] startId=" << mnStartId << "  matchNum=" << matchNum
                  << "  average_parallax=" << average_parallax << std::endl;

        LinkInfo linkInfo;
        linkInfo.mvMatchesBetweenKF = vMatches;
        linkInfo.mvKPIdToMatchedId = vMatchIdx;
        linkInfo.mfAverageParallax = average_parallax;
        linkInfo.mvPoints1 = vPtsUn0;
        linkInfo.mvPoints2 = vPtsUni;
        linkInfo.mnLinkId = static_cast<int>(mvpSlidingWindowKFs.size()) - 1;
        linkInfo.mnMatchedNum = matchNum;
        std::cout << "[Initializer::Insert] startId=" << mnStartId
                  << "  matchId=" << linkInfo.mnLinkId
                  << "  good matchNum =" << matchNum
                  << "  average_parallax=" << average_parallax << std::endl << std::endl;
        mmMatchPairs[mnStartId].emplace_back(linkInfo);

        if(matchNum >= 60 && average_parallax >= 18){
            mnStartId = static_cast<int>(mvpSlidingWindowKFs.size() - 1);
            mStartImg = img.clone();
        }
    }
}

bool Initializer::Initialize(std::vector<KeyFrame*>& vpKFs, std::set<MapPoint*>& spMPs) {
    std::vector<KeyFrame*> vKFs;

    int lastKFId;
    for(auto& matchPair: mmMatchPairs){
        int startId = matchPair.first;
        LinkInfo linkInfo = matchPair.second.back();
        std::cout << "[Initializer::Initialize] startId=" << startId << "  matchedId="
                  << linkInfo.mnLinkId << "  match num=" << linkInfo.mnMatchedNum << std::endl;
        CoarseInit(startId, linkInfo);
        vKFs.emplace_back(mvpSlidingWindowKFs[startId]);
        lastKFId = linkInfo.mnLinkId;
        mpKeyFrameDB->AddKeyFrame(mvpSlidingWindowKFs[startId]);
    }
    vKFs.emplace_back(mvpSlidingWindowKFs[lastKFId]);
    mpKeyFrameDB->AddKeyFrame(mvpSlidingWindowKFs[lastKFId]);
    Optimization::SlidingWindowBA(vKFs, mK);
    int goodMPNum = 0, goodMPNumInKF = 0;
    for(auto* pMP: mspSlidingWindowMPs){
        if(!pMP)
            continue;
        if(pMP->GetObsNum() == 0)
            continue;
        goodMPNum++;
    }
    for(int i = 0; i < mvpSlidingWindowKFs.back()->N; i++){
        if(mvpSlidingWindowKFs.back()->GetMapPoint(i))
            goodMPNumInKF++;
    }
    std::cout << "[Initializer::Initialize] after slidingWindow BA, good mappoints num="
              << goodMPNum << "  good mappoint in lastest KF num=" << goodMPNumInKF << std::endl;

    mpMap->InsertMapPoints(mspSlidingWindowMPs);
    vpKFs = mvpSlidingWindowKFs;
    spMPs = mspSlidingWindowMPs;

    DealWithUnused();
}

void Initializer::CoarseInit(int startId, const LinkInfo& linkInfo) {
    KeyFrame* pStartKF = mvpSlidingWindowKFs[startId];
    auto *pKFi = mvpSlidingWindowKFs[linkInfo.mnLinkId];

    // 第一个匹配对，直接通过基础矩阵恢复位姿，然后三角化
    if(mspSlidingWindowMPs.empty()){
        cv::Mat R21, t21;
        if(SolveRelativePose(linkInfo.mvPoints1, linkInfo.mvPoints2, R21, t21) >= 12){
            pKFi->SetT(R21, t21);
            cv::Mat points4D = Triangulate(linkInfo.mvPoints1, linkInfo.mvPoints2, R21, t21);

            for(int id = 0; id < pStartKF->N; id++){
                int matchedId = linkInfo.mvMatchesBetweenKF[id];
                if(matchedId == -1)
                    continue;

                // 判断重投影误差
                cv::Mat pt3D = points4D.col(linkInfo.mvKPIdToMatchedId[id]);
                pt3D = pt3D.rowRange(0, 3) / pt3D.at<float>(3);

                //判断三维点的深度值是否为正，即是否在相机前方
                if(!CheckPt3DValid(pt3D, pStartKF->GetKeyPointUn(id).pt))
                    continue;

                cv::Mat pt3Di = R21 * pt3D + t21;
                if(!CheckPt3DValid(pt3Di, pKFi->GetKeyPointUn(matchedId).pt))
                    continue;

                cv::Mat description = pStartKF->GetDescription(id);
                auto* pMPNew = new MapPoint(pt3D, pStartKF);
                pMPNew->SetDescription(description);
                pMPNew->AddObservation(pStartKF, id);
                pMPNew->AddObservation(pKFi, matchedId);

                pStartKF->AddMapPoint(id, pMPNew);
                pKFi->AddMapPoint(matchedId, pMPNew);
                mspSlidingWindowMPs.insert(pMPNew);
            }

            std::vector<int> vMatches, vMatchIdx;
            std::vector<cv::Point2f> vPts1, vPts2;
            Matcher::SearchByBow(pStartKF, pKFi, vMatches);
            cv::Mat F12 = pStartKF->ComputeFundamental(pKFi);
            int nBowMatchNum = Matcher::CollectMatches(pStartKF, pKFi, F12, vMatches, vMatchIdx,
                                                       vPts1, vPts2);
            std::cout << "[Initializer::CoarseInit] BoW match num=" << nBowMatchNum << std::endl;
            cv::Mat points4DNew = Triangulate(vPts1, vPts2,
                                              pKFi->GetRotation(), pKFi->GetTranslation());

            std::vector<MapPoint*> vpMPs;
            for(int id = 0; id < pStartKF->N; id++){
                if(vMatchIdx[id] != -1){
                    cv::Mat pt3D = points4DNew.col(vMatchIdx[id]);
                    pt3D = pt3D.rowRange(0, 3) / pt3D.at<float>(3);
                    if(!CheckPt3DValid(pt3D, pStartKF->GetKeyPointUn(id).pt))
                        continue;

                    cv::Mat pt3Di = R21 * pt3D + t21;
                    if(!CheckPt3DValid(pt3Di, pKFi->GetKeyPointUn(vMatches[id]).pt))
                        continue;


                    cv::Mat description = pStartKF->GetDescription(id);
                    auto* pMPNew = new MapPoint(pt3D, pStartKF);
                    pMPNew->SetDescription(description);
                    pMPNew->AddObservation(pStartKF, id);
                    pMPNew->AddObservation(pKFi, vMatches[id]);

                    pStartKF->AddMapPoint(id, pMPNew);
                    pKFi->AddMapPoint(vMatches[id], pMPNew);
                    mspSlidingWindowMPs.insert(pMPNew);
                    vpMPs.emplace_back(pMPNew);
                }
            }
            std::cout << "[Initialize::CoarseInit] mappoint in SW num="
                      << mspSlidingWindowMPs.size() << std::endl;

            DrawPoints(mKFImgs[linkInfo.mnLinkId], pKFi->GetPointsUn(), vpMPs, mK,
                       mDistCoef, pKFi->GetRotation(), pKFi->GetTranslation(), "BoW MP", 2);

            std::vector<KeyFrame*> vKFsForBA{pStartKF, pKFi};
            Optimization::SlidingWindowBA(vKFsForBA, mK);

            DrawPoints(mKFImgs[linkInfo.mnLinkId], pKFi, mK, mDistCoef, "MPProject", 2);
        }
    }
    else{ // 其他匹配对，先通过PnP求位姿，然后三角化其他匹配对
        std::vector<MapPoint*> vpMapPoints(pKFi->N, nullptr);
        for(int id = 0; id < pStartKF->N; id++){
            int matchedId = linkInfo.mvMatchesBetweenKF[id];
            if(matchedId == -1)
                continue;
            MapPoint* pMP = pStartKF->GetMapPoint(id);
            if(!pMP)
                continue;

            vpMapPoints[matchedId] = pMP;
        }

        cv::Mat R21, r21, t21, Tiw, inliers;
        bool bPnP = Optimization::SolvePnP(pKFi->GetPointsUn(), vpMapPoints, mK, Tiw);
        if(bPnP){
            pKFi->SetT(Tiw);
            R21 = pKFi->GetRotation() * pStartKF->GetRotationInv();
            t21 = pKFi->GetTranslation() - R21 * pStartKF->GetTranslation();

            // 对匹配点进行三角化，并检查有效性；都是在局部坐标系中（前一帧的坐标系）
            cv::Mat points4D = Triangulate(linkInfo.mvPoints1, linkInfo.mvPoints2, R21, t21);
            std::cout << "[Initializer::CoarseInit] KLT match num=" << linkInfo.mvPoints1.size() << std::endl;
            int newMPNum = 0, existMPNum=0;
            for(int id = 0; id < pStartKF->N; id++){
                int matchedId = linkInfo.mvMatchesBetweenKF[id];
                if(matchedId == -1)
                    continue;

                MapPoint* pMP = vpMapPoints[matchedId];
                if(pMP){
                    if(!pMP->IsBad()) {
                        pMP->AddObservation(pKFi, matchedId);
                        pKFi->AddMapPoint(matchedId, pMP);
                        existMPNum++;
                    }
                }
                else {
                    // 判断三角化的点在两帧上的重投影误差
                    cv::Mat pt3D = points4D.col(linkInfo.mvKPIdToMatchedId[id]);
                    pt3D = pt3D.rowRange(0, 3) / pt3D.at<float>(3);
                    if(!CheckPt3DValid(pt3D, pStartKF->GetKeyPointUn(id).pt))
                        continue;

                    cv::Mat pt3Di = R21 * pt3D + t21;
                    if(!CheckPt3DValid(pt3Di, pKFi->GetKeyPointUn(matchedId).pt))
                        continue;

                    cv::Mat pt3DW = pStartKF->GetRotationInv() * pt3D + pStartKF->GetCameraCenter();
                    // 转换到初始帧的坐标系中。从前一帧的相机坐标系转换到世界坐标系
                    cv::Mat description = pStartKF->GetDescription(id);
                    if(!pStartKF->GetMapPoint(id)) {
                        auto *pMPNew = new MapPoint(pt3DW, pStartKF); // 创建地图点，并设置坐标和参考关键帧
                        pMPNew->SetDescription(description);
                        pMPNew->AddObservation(pStartKF, id);
                        pMPNew->AddObservation(pKFi, matchedId);

                        pStartKF->AddMapPoint(id, pMPNew);
                        pKFi->AddMapPoint(matchedId, pMPNew);
                        vpMapPoints[matchedId] = pMPNew;
//                        mspSlidingWindowMPs.insert(pMPNew);
                        newMPNum++;
                    }
                    else{
                        // pStartKF中存在pMP，说明这个点是PnP求解中的outlier。
                        continue;
                    }
                }
            }
            std::cout << "[Initializer::CoarseInit] newMPNum=" << newMPNum
                      << " existMPNum=" << existMPNum << std::endl;
        }
        else{
            if(SolveRelativePose(linkInfo.mvPoints1, linkInfo.mvPoints2, R21, t21) >= 12){
                cv::Mat points4D = Triangulate(linkInfo.mvPoints1, linkInfo.mvPoints2, R21, t21);
                cv::Mat Riw, tiw;
                Riw = R21 * pStartKF->GetRotation();
                tiw = R21 * pStartKF->GetTranslation() + t21;
                pKFi->SetT(Riw, tiw);

                int negNum=0;
                for(int id = 0; id < pStartKF->N; id++){
                    int matchedId = linkInfo.mvMatchesBetweenKF[id];
                    if(matchedId == -1)
                        continue;

                    MapPoint* pMP = pStartKF->GetMapPoint(id);
                    if(pMP){
                        if(!pMP->IsBad()) {
                            pMP->AddObservation(pKFi, matchedId);
                            pKFi->AddMapPoint(matchedId, pMP);
                        }
                    }
                    else {
                        // 对于不存在的mappoint，创建新的MapPoint
                        // 此处是否应该判断一下重投影误差?
                        cv::Mat pt3D = points4D.col(linkInfo.mvKPIdToMatchedId[id]);
                        pt3D = pt3D.rowRange(0, 3) / pt3D.at<float>(3);

                        if(!CheckPt3DValid(pt3D, pStartKF->GetKeyPointUn(id).pt))
                            continue;

                        cv::Mat pt3Di = R21 * pt3D + t21;
                        if(!CheckPt3DValid(pt3Di, pKFi->GetKeyPointUn(matchedId).pt))
                            continue;

                        cv::Mat pt3DW = pStartKF->GetRotationInv() * pt3D + pStartKF->GetCameraCenter();
                        cv::Mat description = pStartKF->GetDescription(id);
                        auto *pMPNew = new MapPoint(pt3DW, pStartKF);
                        pMPNew->SetDescription(description);
                        pMPNew->AddObservation(pStartKF, id);
                        pMPNew->AddObservation(pKFi, matchedId);

                        pStartKF->AddMapPoint(id, pMPNew);
                        pKFi->AddMapPoint(matchedId, pMPNew);
                        vpMapPoints[matchedId] = pMPNew;
//                        mspSlidingWindowMPs.insert(pMPNew);
                    }
                }
                std::cout << "[Initializer::CoarseInit] pt3D neg z num=" << negNum << std::endl;
            }
        }

        // vMatches:两帧中关键点id之间的匹配。
        // vMatchIdx: 前一帧关键点id对应的重新排列后的下标id
        std::vector<int> vMatches, vMatchIdx;
        std::vector<cv::Point2f> vPts1, vPts2;
        Matcher::SearchByBow(pStartKF, pKFi, vMatches);
        cv::Mat F12 = pStartKF->ComputeFundamental(pKFi);
        int nBowMatchNum = Matcher::CollectMatches(pStartKF, pKFi, F12, vMatches, vMatchIdx,
                                                   vPts1, vPts2);
        std::cout << "[Initializer::CoarseInit] BoW match num=" << nBowMatchNum << std::endl;
        cv::Mat points4DNew = Triangulate(vPts1, vPts2, R21, t21);
        for(int id = 0; id < pStartKF->N; id++){
            if(vMatchIdx[id] != -1){
                cv::Mat pt3D = points4DNew.col(vMatchIdx[id]);
                pt3D = pt3D.rowRange(0, 3) / pt3D.at<float>(3);

                if(!CheckPt3DValid(pt3D, pStartKF->GetKeyPointUn(id).pt))
                    continue;

                cv::Mat pt3Di = R21 * pt3D + t21;
                if(!CheckPt3DValid(pt3Di, pKFi->GetKeyPointUn(vMatches[id]).pt))
                    continue;

                cv::Mat pt3DW = pStartKF->GetRotationInv() * pt3D + pStartKF->GetCameraCenter();
                cv::Mat description = pStartKF->GetDescription(id);
                auto* pMPNew = new MapPoint(pt3DW, pStartKF);
                pMPNew->SetDescription(description);
                pMPNew->AddObservation(pStartKF, id);
                pMPNew->AddObservation(pKFi, vMatches[id]);

                pStartKF->AddMapPoint(id, pMPNew);
                pKFi->AddMapPoint(vMatches[id], pMPNew);
                vpMapPoints[vMatches[id]] = pMPNew;
            }
        }
        DrawPoints(mKFImgs[linkInfo.mnLinkId], pKFi->GetPointsUn(), vpMapPoints, mK,
                   mDistCoef, pKFi->GetRotation(), pKFi->GetTranslation(), "BoW MP", 2);
        std::cout << "MapPoints Num in SW after BoW match: " << mspSlidingWindowMPs.size() << std::endl;

        Tiw = pKFi->GetTcw();
        std::vector<bool> vOutlier;
        std::vector<float> vChi2;

        Optimization::PoseOptimize(pKFi->GetKeyPointsUn(), vpMapPoints, pKFi->GetInvLevelSigma2(),
                                   mK, Tiw, vOutlier, vChi2);
        pKFi->SetT(Tiw);

        std::cout << "MapPoints Num in SW before: " << mspSlidingWindowMPs.size() << std::endl;
        int outlierNum = 0;
        for(int i = 0; i < vOutlier.size(); i++){
            MapPoint* pMP = vpMapPoints[i];
            if(!pMP)
                continue;
            if(vOutlier[i]){
                vpMapPoints[i] = nullptr;
                pKFi->EraseMapPoint(pMP);
                pMP->EraseObservation(pKFi);
                outlierNum++;
                continue;
            }
            mspSlidingWindowMPs.insert(pMP);
        }

        int MPNum = 0;
        for(int i = 0; i < pKFi->N; i++){
            MapPoint* pMP = pKFi->GetMapPoint(i);
            if(pMP)
                MPNum++;
        }
        std::cout << "MP num=" << MPNum << std::endl;
        std::cout << "Outlier num=" << outlierNum << std::endl;
        std::cout << "MapPoints Num in SW after: " << mspSlidingWindowMPs.size() << std::endl;
        DrawPoints(mKFImgs[linkInfo.mnLinkId], pKFi, mK, mDistCoef, "MPProject", 2, vChi2);

    }
}

int Initializer::SolveRelativePose(const vector<cv::Point2f> &vPts0,
                                   const vector<cv::Point2f> &vPts1,
                                   cv::Mat &R10, cv::Mat &t10) {
    cv::Mat mask;
//    cv::Mat EMat = cv::findEssentialMat(vPts0, vPts1, mK, cv::RANSAC, 0.999, 5.99, mask);
    cv::Mat EMat = cv::findEssentialMat(vPts0, vPts1, mK, cv::RANSAC, 0.999, 3.84, mask);
    int inlier_cnt = cv::recoverPose(EMat, vPts0, vPts1, mK, R10, t10, mask);
    R10.convertTo(R10, CV_32F);
    t10.convertTo(t10, CV_32F);
    std::cout << "[Initializer::SolveRelativePose] inlier_cnt=" << inlier_cnt << std::endl;
    return inlier_cnt;
}

cv::Mat Initializer::Triangulate(const vector<cv::Point2f> &vPts1, const vector<cv::Point2f> &vPts2,
                         const cv::Mat &R10, const cv::Mat &t10) {
    cv::Mat points4D;
    cv::Mat P1(3, 4, CV_32F, cv::Scalar(0));
    mK.copyTo(P1.rowRange(0, 3).colRange(0, 3));
    cv::Mat P2(3, 4, CV_32F, cv::Scalar(0));
    R10.copyTo(P2.rowRange(0, 3).colRange(0, 3));
    t10.copyTo(P2.rowRange(0, 3).col(3));
    P2 = mK * P2;
    cv::triangulatePoints(P1, P2, vPts1, vPts2, points4D);
    return points4D;
}

cv::Mat Initializer::TransformPoints(const cv::Mat& points4D, const cv::Mat& T){
    cv::Mat points4DNew;
    points4DNew = T * points4D;
    return points4DNew;
}

cv::Point2f Initializer::project(const cv::Mat &pt3d) const {
    float x_norm = pt3d.at<float>(0) / pt3d.at<float>(2);
    float y_norm = pt3d.at<float>(1) / pt3d.at<float>(2);
    float x_un = x_norm * mK.at<float>(0, 0) + mK.at<float>(0, 2);
    float y_un = y_norm * mK.at<float>(1, 1) + mK.at<float>(1, 2);
    return {x_un, y_un};
}

bool Initializer::ReadyToInit() {
    if (mvpSlidingWindowKFs.size() == mnSlidingWindowSize)
        return true;
    else
        return false;
}

void Initializer::DealWithUnused() {
    for(auto& matchPair: mmMatchPairs){
        int startId = matchPair.first;
        std::vector<LinkInfo> linkInfos = matchPair.second;
        if(linkInfos.size() == 1)
            continue;
        KeyFrame* pStartKF = mvpSlidingWindowKFs[startId];
        for(int i = 0; i < linkInfos.size() - 1; i++){
            LinkInfo linkInfo = linkInfos[i];
            KeyFrame* pKF = mvpSlidingWindowKFs[linkInfo.mnLinkId];
            std::vector<MapPoint*> vMPs;
            std::vector<cv::Point3f> vPts3D;
            std::vector<cv::Point2f> vPts2D;
            std::vector<cv::KeyPoint> vKPsUn;
            std::vector<int> vMatchedIdx;
            for(int id = 0; id < linkInfo.mvMatchesBetweenKF.size(); id++){
                if(linkInfo.mvMatchesBetweenKF[id] == -1)
                    continue;
                MapPoint* pMP = pStartKF->GetMapPoint(id);
                if(!pMP)
                    continue;
                vMPs.emplace_back(pMP);
                cv::Mat mpPos = pMP->GetWorldPos();
                vPts3D.emplace_back(cv::Point3f(mpPos.at<float>(0), mpPos.at<float>(1), mpPos.at<float>(2)));
                vPts2D.emplace_back(pKF->GetKeyPointUn(linkInfo.mvMatchesBetweenKF[id]).pt);
                vKPsUn.emplace_back(pKF->GetKeyPointUn(linkInfo.mvMatchesBetweenKF[id]));
                vMatchedIdx.emplace_back(linkInfo.mvMatchesBetweenKF[id]);
            }
            cv::Mat R21, r21, t21, inliers;
            if(vPts3D.size() < 4){
                std::cout << "[Initializer::DealWithUnused] points num for PNP=" << vPts3D.size() << std::endl;
                exit(0);
            }
            cv::solvePnPRansac(vPts3D, vPts2D, mK, cv::Mat::zeros(4, 1, CV_32F),
                               r21, t21, false, 100, 4, 0.99, inliers);
            cv::Rodrigues(r21, R21);
            R21.convertTo(R21, CV_32F);
            t21.convertTo(t21, CV_32F);
            cv::Mat Ri0, ti0;
            Ri0 = R21 * pStartKF->GetRotation();
            ti0 = R21 * pStartKF->GetTranslation() + t21;
            pKF->SetT(Ri0, ti0);
            cv::Mat Tcw = pKF->GetTcw();
            std::vector<bool> vOutlier;
            std::vector<float> vChi2;
            std::cout << "[Initializer::DealWithUnused] MapPoints match num before g2o: " << vMPs.size() << std::endl;
            int inliers_num = Optimization::PoseOptimize(vKPsUn, vMPs, pKF->GetInvLevelSigma2(), mK, Tcw, vOutlier, vChi2);
            std::cout << "[Initializer::DealWithUnused] MapPoints match num after g2o: " << inliers_num << std::endl;
            pKF->SetT(Tcw);
            for(int j = 0; j < vOutlier.size(); j++){
                if(!vOutlier[j]){
                    MapPoint* pMP = vMPs[j];
                    pKF->AddMapPoint(vMatchedIdx[j], pMP);
                }
            }
        }
    }
}

void Initializer::InitBetweenKFs(vector<KeyFrame *> &vKFs) {
    std::chrono::system_clock::time_point t1 = std::chrono::system_clock::now();
    for (int i = 0; i < vKFs.size() - 1; i++){
        KeyFrame* pStartKF = vKFs[i];
        for (int j = i + 1; j < vKFs.size(); j++){
            KeyFrame* pKF = vKFs[j];
            std::vector<int> vMatches;
            int matchNum = Matcher::SearchByBow(pStartKF, pKF, vMatches);
            std::cout << "[Initializer::InitBetweenKFs] startId=" << i << "  curId=" << j
                      << "  BoW match num=" << matchNum << std::endl;
            std::vector<cv::Point2f> vPts1, vPts2;
            std::vector<int> vIdxInKF1, vIdxInKF2;
            vPts1.reserve(matchNum);
            vPts2.reserve(matchNum);
            for(int id = 0; id < pStartKF->N; id++){
                if(vMatches[id] == -1)
                    continue;
                vPts1.emplace_back(pStartKF->GetKeyPointUn(id).pt);
                vPts2.emplace_back(pKF->GetKeyPointUn(vMatches[id]).pt);
                vIdxInKF1.emplace_back(id);
                vIdxInKF2.emplace_back(vMatches[id]);
            }
            cv::Mat R21, t21;
            R21 = pKF->GetRotation() * pStartKF->GetRotationInv();
            t21 = pKF->GetTranslation() - R21 * pStartKF->GetTranslation();
            cv::Mat points4D = Triangulate(vPts1, vPts2, R21, t21);
            TransformPoints(points4D, pStartKF->GetTwc());

            int negNum = 0;
            for (int k = 0; k < matchNum; k++){
                cv::Mat pt3D = points4D.col(k);
                pt3D = pt3D.rowRange(0, 3) / pt3D.at<float>(3);
                if(pt3D.at<float>(2) <= 0) {
                    negNum++;
                    continue;
                }

                cv::Mat description = pStartKF->GetDescription(vIdxInKF1[k]);
                auto *pMPNew = new MapPoint(pt3D, pStartKF);
                pMPNew->SetDescription(description);
                pMPNew->AddObservation(pStartKF, vIdxInKF1[k]);
                pMPNew->AddObservation(pKF, vIdxInKF2[k]);

                pStartKF->AddMapPoint(vIdxInKF1[k], pMPNew);
                pKF->AddMapPoint(vIdxInKF2[k], pMPNew);
                mspSlidingWindowMPs.insert(pMPNew);
            }
        }
    }

}

bool Initializer::CheckPt3DValid(const cv::Mat& pt3D, const cv::Point2f& ptUn) {
    if(pt3D.at<float>(2) <= 0)
        return false;
    cv::Point2f ptProj = project(pt3D);
    float dist = sqrt((ptProj.x - ptUn.x) * (ptProj.x - ptUn.x) +
                      (ptProj.y - ptUn.y) * (ptProj.y - ptUn.y));
    if (dist > 2)
        return false;
    return true;
}

} // namespace Naive_SLAM