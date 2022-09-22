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
//    InitBetweenKFs(vKFs);
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

//    DrawPoints(mStartImg, mvpSlidingWindowKFs.back()->GetPointsUn(),
//               mvpSlidingWindowKFs.back()->GetMapPoints(), mK, mDistCoef,
//               mvpSlidingWindowKFs.back()->GetRotation(),
//               mvpSlidingWindowKFs.back()->GetTranslation(), "KeyFrame");

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

            int negNum = 0;
            for(int id = 0; id < pStartKF->N; id++){
                if(linkInfo.mvMatchesBetweenKF[id] == -1)
                    continue;

                // 判断重投影误差
                cv::Mat pt3D = points4D.col(linkInfo.mvKPIdToMatchedId[id]);
                pt3D = pt3D.rowRange(0, 3) / pt3D.at<float>(3);

                //判断三维点的深度值是否为正，即是否在相机前方
                if(pt3D.at<float>(2) <= 0) {
                    negNum++;
                    continue;
                }

                cv::Point2f ptProj = project(pt3D);
                cv::Point2f pt = pStartKF->GetKeyPointUn(id).pt;
                float dist = sqrt((ptProj.x - pt.x) * (ptProj.x - pt.x) +
                                  (ptProj.y - pt.y) * (ptProj.y - pt.y));
                if (dist > 2) continue;

                cv::Mat pt3Di = R21 * pt3D + t21;
                ptProj = project(pt3Di);
                pt = pKFi->GetKeyPointUn(linkInfo.mvMatchesBetweenKF[id]).pt;
                dist = sqrt((ptProj.x - pt.x) * (ptProj.x - pt.x) +
                            (ptProj.y - pt.y) * (ptProj.y - pt.y));
                if (dist > 2) continue;

                cv::Mat description = pStartKF->GetDescription(id);
                auto* mapPoint = new MapPoint(pt3D, pStartKF);
                mapPoint->SetDescription(description);
                mapPoint->AddObservation(pStartKF, id);
                mapPoint->AddObservation(pKFi, linkInfo.mvMatchesBetweenKF[id]);

                pStartKF->AddMapPoint(id, mapPoint);
                pKFi->AddMapPoint(linkInfo.mvMatchesBetweenKF[id], mapPoint);
                mspSlidingWindowMPs.insert(mapPoint);
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

            negNum = 0;
            std::vector<MapPoint*> vpMPs;
            for(int id = 0; id < pStartKF->N; id++){
                if(vMatchIdx[id] != -1){
                    cv::Mat pt3D = points4DNew.col(vMatchIdx[id]);
                    pt3D = pt3D.rowRange(0, 3) / pt3D.at<float>(3);
                    if(pt3D.at<float>(2) <= 0) {
                        negNum++;
                        continue;
                    }
                    cv::Mat description = pStartKF->GetDescription(id);
                    auto* mapPoint = new MapPoint(pt3D, pStartKF);
                    mapPoint->SetDescription(description);
                    mapPoint->AddObservation(pStartKF, id);
                    mapPoint->AddObservation(pKFi, vMatches[id]);

                    pStartKF->AddMapPoint(id, mapPoint);
                    pKFi->AddMapPoint(vMatches[id], mapPoint);
                    mspSlidingWindowMPs.insert(mapPoint);
                    vpMPs.emplace_back(mapPoint);
                }
            }
            std::cout << "[Initializer::CoarseInit] BoW neg z num=" << negNum << std::endl;
            std::cout << "[Initialize::CoarseInit] mappoint in SW num="
                      << mspSlidingWindowMPs.size() << std::endl;

            DrawPoints(mKFImgs[linkInfo.mnLinkId], pKFi->GetPointsUn(), vpMPs, mK,
                       mDistCoef, pKFi->GetRotation(), pKFi->GetTranslation(), "BoW MP", 2);

            PrintMat("pKFi", pKFi->GetTcw());
            std::cout << "[Initializer::CoarseInit] pt3D neg z num=" << negNum << std::endl;
            std::vector<KeyFrame*> vKFsForBA{pStartKF, pKFi};
            Optimization::SlidingWindowBA(vKFsForBA, mK);

            DrawPoints(mKFImgs[linkInfo.mnLinkId], pKFi, mK, mDistCoef, "MPProject", 2);
        }
    }
    else{ // 其他匹配对，先通过PnP求位姿，然后三角化其他匹配对
        std::vector<cv::Point3f> vPts3D;
        std::vector<cv::Point2f> vPts2D;
        for(int id = 0; id < pStartKF->N; id++){
            int matchedId = linkInfo.mvMatchesBetweenKF[id];
            if(matchedId == -1)
                continue;
            MapPoint* pMP = pStartKF->GetMapPoint(id);
            if(!pMP)
                continue;

            // 把地图点转换到StartKF的相机坐标系中
            cv::Mat worldPos = pMP->GetWorldPos();
            cv::Mat startPos = pStartKF->GetRotation() * worldPos + pStartKF->GetTranslation();
            vPts3D.emplace_back(cv::Point3f(startPos.at<float>(0), startPos.at<float>(1), startPos.at<float>(2)));
            cv::Point2f matchedPt2D = pKFi->GetKeyPointUn(matchedId).pt;
            vPts2D.emplace_back(matchedPt2D);
        }

        cv::Mat R21, r21, t21, inliers;
        // 先尝试pnp
        std::cout << "[Initializer::CoarseInit] 3D 2D match num=" << vPts3D.size() << std::endl;
        if(!vPts3D.empty() && !vPts2D.empty()) {
            // 局部坐标系下，第二帧相对于第一帧的位姿
            cv::solvePnPRansac(vPts3D, vPts2D, mK, cv::Mat::zeros(4, 1, CV_32F),
                               r21, t21, false, 100, 4, 0.99, inliers, cv::SOLVEPNP_EPNP);
            std::cout << "[Initializer::CoarseInit] matches with MapPoint num=" << vPts2D.size()
                      << "  PnP inliers num=" << inliers.total() << std::endl;
        }
        if(!inliers.empty() && inliers.total() >= 10){
            cv::Rodrigues(r21, R21);
            R21.convertTo(R21, CV_32F);
            t21.convertTo(t21, CV_32F);

            // 根据pnp求出的当前帧的初始位姿
            cv::Mat Ri0, ti0;
            Ri0 = R21 * pStartKF->GetRotation();
            ti0 = R21 * pStartKF->GetTranslation() + t21;
            pKFi->SetT(Ri0, ti0);

            // 对匹配点进行三角化，并检查有效性；都是在局部坐标系中（前一帧的坐标系）
            cv::Mat points4D = Triangulate(linkInfo.mvPoints1, linkInfo.mvPoints2, R21, t21);
            std::cout << "[Initializer::CoarseInit] KLT match num=" << linkInfo.mvPoints1.size() << std::endl;
            int newMPNum = 0, existMPNum=0;
            int negNum = 0;
            for(int id = 0; id < pStartKF->N; id++){
                if(linkInfo.mvMatchesBetweenKF[id]==-1)
                    continue;

                MapPoint* pMP = pStartKF->GetMapPoint(id);
                if(pMP){
                    if(!pMP->IsBad()) {
                        pMP->AddObservation(pKFi, linkInfo.mvMatchesBetweenKF[id]);
                        pKFi->AddMapPoint(linkInfo.mvMatchesBetweenKF[id], pMP);
                        existMPNum++;
                    }
                }
                else {
                    // 判断三角化的点在两帧上的重投影误差
                    cv::Mat pt3D = points4D.col(linkInfo.mvKPIdToMatchedId[id]);
                    pt3D = pt3D.rowRange(0, 3) / pt3D.at<float>(3);

                    if(pt3D.at<float>(2) <= 0) {
                        negNum++;
                        continue;
                    }

                    cv::Point2f ptProj = project(pt3D);
                    cv::Point2f pt = pStartKF->GetKeyPointUn(id).pt;
                    float dist = sqrt((ptProj.x - pt.x) * (ptProj.x - pt.x) +
                                      (ptProj.y - pt.y) * (ptProj.y - pt.y));
                    if (dist > 2) continue;

                    cv::Mat pt3Di = R21 * pt3D + t21;
                    ptProj = project(pt3Di);
                    pt = pKFi->GetKeyPointUn(linkInfo.mvMatchesBetweenKF[id]).pt;
                    dist = sqrt((ptProj.x - pt.x) * (ptProj.x - pt.x) +
                                (ptProj.y - pt.y) * (ptProj.y - pt.y));
                    if (dist > 2) continue;

                    // 转换到初始帧的坐标系中。从前一帧的相机坐标系转换到世界坐标系
                    cv::Mat pt3D0 = pStartKF->GetRotationInv() * pt3D + pStartKF->GetCameraCenter();

                    cv::Mat description = pStartKF->GetDescription(id);
                    auto *pMPNew = new MapPoint(pt3D0, pStartKF); // 创建地图点，并设置坐标和参考关键帧
                    pMPNew->SetDescription(description);
                    pMPNew->AddObservation(pStartKF, id);
                    pMPNew->AddObservation(pKFi, linkInfo.mvMatchesBetweenKF[id]);

                    pStartKF->AddMapPoint(id, pMPNew);
                    pKFi->AddMapPoint(linkInfo.mvMatchesBetweenKF[id], pMPNew);
                    mspSlidingWindowMPs.insert(pMPNew);
                    newMPNum++;
                }
            }
            std::cout << "[Initializer::CoarseInit] pt3D neg z num=" << negNum << std::endl;
            std::cout << "[Initializer::CoarseInit] newMPNum=" << newMPNum
                      << " existMPNum=" << existMPNum << std::endl;
        }
        else{
            if(SolveRelativePose(linkInfo.mvPoints1, linkInfo.mvPoints2, R21, t21) >= 12){
                cv::Mat points4D = Triangulate(linkInfo.mvPoints1, linkInfo.mvPoints2, R21, t21);
                cv::Mat Ri0, ti0;
                Ri0 = R21 * pStartKF->GetRotation();
                ti0 = R21 * pStartKF->GetTranslation() + t21;
                pKFi->SetT(Ri0, ti0);
                // 三角化后的点转换到世界坐标系中
//                points4D = TransformPoints(points4D, pKFi->GetTwc());
//                cv::Mat points4DW = TransformPoints(points4D, pStartKF->GetTwc());

                int negNum=0;
                for(int id = 0; id < pStartKF->N; id++){
                    if(linkInfo.mvMatchesBetweenKF[id] == -1)
                        continue;

                    MapPoint* pMP = pStartKF->GetMapPoint(id);
                    if(pMP){
                        if(!pMP->IsBad()) {
                            pMP->AddObservation(pKFi, linkInfo.mvMatchesBetweenKF[id]);
                            pKFi->AddMapPoint(linkInfo.mvMatchesBetweenKF[id], pMP);
                        }
                    }
                    else {
                        // 对于不存在的mappoint，创建新的MapPoint
                        // 此处是否应该判断一下重投影误差?
                        cv::Mat pt3D = points4D.col(linkInfo.mvKPIdToMatchedId[id]);
                        pt3D = pt3D.rowRange(0, 3) / pt3D.at<float>(3);

                        if (pt3D.at<float>(2) <= 0){
                            negNum++;
                            continue;
                        }

                        cv::Point2f ptProj = project(pt3D);
                        cv::Point2f pt = pStartKF->GetKeyPointUn(id).pt;
                        float dist = sqrt((ptProj.x - pt.x) * (ptProj.x - pt.x) +
                                          (ptProj.y - pt.y) * (ptProj.y - pt.y));
                        if (dist > 2) continue;

                        cv::Mat pt3Di = R21 * pt3D + t21;
                        ptProj = project(pt3Di);
                        pt = pKFi->GetKeyPointUn(linkInfo.mvMatchesBetweenKF[id]).pt;
                        dist = sqrt((ptProj.x - pt.x) * (ptProj.x - pt.x) +
                                    (ptProj.y - pt.y) * (ptProj.y - pt.y));
                        if (dist > 2) continue;

                        cv::Mat pt3D0 = pStartKF->GetRotationInv() * pt3D + pStartKF->GetCameraCenter();
                        cv::Mat description = pStartKF->GetDescription(id);
                        auto *mapPoint = new MapPoint(pt3D0, pStartKF);
                        mapPoint->SetDescription(description);
                        mapPoint->AddObservation(pStartKF, id);
                        mapPoint->AddObservation(pKFi, linkInfo.mvMatchesBetweenKF[id]);

                        pStartKF->AddMapPoint(id, mapPoint);
                        pKFi->AddMapPoint(linkInfo.mvMatchesBetweenKF[id], mapPoint);
                        mspSlidingWindowMPs.insert(mapPoint);
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
        cv::Mat R21_tmp = pKFi->GetRotation() * pStartKF->GetRotationInv();
        cv::Mat t21_tmp = pKFi->GetTranslation() - R21_tmp * pStartKF->GetTranslation();
        cv::Mat points4DNew = Triangulate(vPts1, vPts2, R21, t21);
        cv::Mat points4DW = TransformPoints(points4DNew, pStartKF->GetTwc());
        int negNum = 0;
        std::vector<MapPoint*> vpMPs;
        for(int id = 0; id < pStartKF->N; id++){
            if(vMatchIdx[id] != -1){
                cv::Mat pt3D = points4DNew.col(vMatchIdx[id]);
                pt3D = pt3D.rowRange(0, 3) / pt3D.at<float>(3);
                if(pt3D.at<float>(2) <= 0) {
                    negNum++;
                    continue;
                }
                pt3D = points4DW.col(vMatchIdx[id]);
                pt3D = pt3D.rowRange(0, 3) / pt3D.at<float>(3);
                cv::Mat description = pStartKF->GetDescription(id);
                auto* mapPoint = new MapPoint(pt3D, pStartKF);
                mapPoint->SetDescription(description);
                mapPoint->AddObservation(pStartKF, id);
                mapPoint->AddObservation(pKFi, vMatches[id]);

                pStartKF->AddMapPoint(id, mapPoint);
                pKFi->AddMapPoint(vMatches[id], mapPoint);
                mspSlidingWindowMPs.insert(mapPoint);
                vpMPs.emplace_back(mapPoint);
            }
        }
        DrawPoints(mKFImgs[linkInfo.mnLinkId], pKFi->GetPointsUn(), vpMPs, mK,
                   mDistCoef, pKFi->GetRotation(), pKFi->GetTranslation(), "BoW MP", 2);
        std::cout << "[Initializer::CoarseInit] pt3D neg z num=" << negNum << std::endl;
        std::cout << "MapPoints Num in SW after BoW match: " << mspSlidingWindowMPs.size() << std::endl;

        // 初始R、t得到后，用g2o进行优化
        std::vector<cv::KeyPoint> vKPsUn;
        std::vector<MapPoint*> vMPs;
        for (int i = 0; i < pKFi->N; i++){
            MapPoint* pMP = pKFi->GetMapPoint(i);
            if(pMP){
                vMPs.emplace_back(pMP);
                vKPsUn.emplace_back(pKFi->GetKeyPointUn(i));
            }
        }
        cv::Mat Tcw = pKFi->GetTcw();
        PrintMat("Tcw before g2o:", Tcw);
        std::vector<bool> vOutlier;
        std::vector<float> vChi2;
        std::cout << "[Initializer::CoarseInit] MapPoints num before g2o: " << vMPs.size()
                  << " corr points num: " << vKPsUn.size() << std::endl;

        // vOutlier与vMPs的顺序对应，与帧中的关键点顺序不对应
        int inliers_num = Optimization::PoseOptimize(vKPsUn, vMPs, pKFi->GetInvLevelSigma2(),
                                                     mK, Tcw, vOutlier, vChi2);
        std::cout << "[Initializer::CoarseInit] MapPoints num after g2o: " << inliers_num << std::endl;
        pKFi->SetT(Tcw);
        PrintMat("Tcw after g2o:", Tcw);

        std::cout << "MapPoints Num in SW before: " << mspSlidingWindowMPs.size() << std::endl;
        int outlierNum = 0;
        for(int i = 0; i < vOutlier.size(); i++){
            MapPoint* pMP = vMPs[i];
            if(vOutlier[i]/* || vChi2[i] > 4*/){
                mspSlidingWindowMPs.erase(pMP);
                pKFi->EraseMapPoint(pMP);
                pMP->EraseObservation(pKFi);
                outlierNum++;
                std::cout << "outlier id=" << i << " chi=" << vChi2[i] << std::endl;
            }
        }
        int MPNum = 0;
        for(int i = 0; i < pKFi->N; i++){
            MapPoint* pMP = pKFi->GetMapPoint(i);
            if(pMP/* && !pMP->IsBad()*/)
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
                auto *mapPoint = new MapPoint(pt3D, pStartKF);
                mapPoint->SetDescription(description);
                mapPoint->AddObservation(pStartKF, vIdxInKF1[k]);
                mapPoint->AddObservation(pKF, vIdxInKF2[k]);

                pStartKF->AddMapPoint(vIdxInKF1[k], mapPoint);
                pKF->AddMapPoint(vIdxInKF2[k], mapPoint);
                mspSlidingWindowMPs.insert(mapPoint);
            }
        }
    }

}

} // namespace Naive_SLAM