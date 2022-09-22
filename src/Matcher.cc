//
// Created by hanfuyong on 2022/8/30.
//

#include "Matcher.h"

namespace Naive_SLAM {

    int Matcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b) {
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

    int Matcher::SearchByBow(Naive_SLAM::KeyFrame *pKF1, Naive_SLAM::KeyFrame *pKF2,
                             std::vector<int> &matches) {
        int matchNum = 0;

        pKF1->ComputeBow();
        pKF2->ComputeBow();

        const DBoW2::FeatureVector &vFeatVec1 = pKF1->GetFeatVec();
        const DBoW2::FeatureVector &vFeatVec2 = pKF2->GetFeatVec();
        auto f1it = vFeatVec1.begin();
        auto f2it = vFeatVec2.begin();
        auto f1end = vFeatVec1.end();
        auto f2end = vFeatVec2.end();

        matches.resize(pKF1->N, -1);
        std::vector<bool> vbMatched2(pKF2->N, false);
        while (f1it != f1end && f2it != f2end) {
            if (f1it->first == f2it->first) {
                for (size_t i1 = 0; i1 < f1it->second.size(); i1++) {
                    const int idx1 = static_cast<int>(f1it->second[i1]);
                    MapPoint *pMP1 = pKF1->GetMapPoint(idx1);
//                    if(pKF1->GetKeyPoint(idx1).octave != 0) continue;
                    if (pMP1 && !pMP1->IsBad())
                        continue;

                    cv::Mat description1 = pKF1->GetDescription(idx1);

                    int bestDist = 50;
                    int bestIdx = -1;
                    for (size_t i2 = 0; i2 < f2it->second.size(); i2++) {
                        const int idx2 = static_cast<int>(f2it->second[i2]);
                        MapPoint *pMP2 = pKF2->GetMapPoint(idx2);
//                        if(pKF2->GetKeyPoint(idx2).octave != 0) continue;
                        if (pMP2 && !pMP2->IsBad() && !vbMatched2[idx2])
                            continue;

                        cv::Mat description2 = pKF2->GetDescription(idx2);
                        int dist = DescriptorDistance(description1, description2);
                        if (dist < bestDist) {
                            bestDist = dist;
                            bestIdx = idx2;
                        }
                    }
                    if (bestIdx >= 0) {
                        matches[idx1] = bestIdx;
                        vbMatched2[bestIdx] = true;
                        matchNum++;
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
        return matchNum;
    }

int Matcher::CollectMatches(const Naive_SLAM::KeyFrame *pKF1, const Naive_SLAM::KeyFrame *pKF2,
                                 std::vector<int> &vMatches, std::vector<int> &vMatchIdx,
                                 std::vector<cv::Point2f>& vPts1, std::vector<cv::Point2f>& vPts2,
                                 float& average_parallax,
                                 std::vector<cv::Point2f>& vPts1Tmp, std::vector<cv::Point2f>& vPts2Tmp) {
    float sum_parallax = 0;
    int idx = 0;
    vMatchIdx.resize(pKF1->N, -1);
    int bad_parallax_num = 0;
    std::vector<float> parallaxs;
    parallaxs.reserve(pKF1->N);
    for(int k = 0; k < pKF1->N; k++) {
        if (vMatches[k] != -1) {
            cv::Point2f pt1 = pKF1->GetKeyPointUn(k).pt;
            cv::Point2f pt2 = pKF2->GetKeyPointUn(vMatches[k]).pt;
            auto parallax = static_cast<float>(sqrt((pt1.x - pt2.x) * (pt1.x - pt2.x) +
                                                    (pt1.y - pt2.y) * (pt1.y - pt2.y)));
            parallaxs.emplace_back(parallax);
        }
    }
    std::sort(parallaxs.begin(), parallaxs.end());
    float medium_parallax = parallaxs[parallaxs.size() / 2];
    std::cout << "[Matcher::CollectMatches] medium_parallax=" << medium_parallax << std::endl;

    for(int k = 0; k < pKF1->N; k++){
        if(vMatches[k] != -1){
            cv::Point2f pt1 = pKF1->GetKeyPointUn(k).pt;
            cv::Point2f pt2 = pKF2->GetKeyPointUn(vMatches[k]).pt;
            auto parallax = static_cast<float>(sqrt((pt1.x - pt2.x) * (pt1.x - pt2.x) +
                                                    (pt1.y - pt2.y) * (pt1.y - pt2.y)));
            if(parallax >= medium_parallax * 3){ // 两个匹配像素的距离太大，认为是误匹配
                vMatches[k] = -1;
                bad_parallax_num++;
                continue;
            }
            sum_parallax += parallax;
            vPts1.emplace_back(pt1);
            vPts2.emplace_back(pt2);
            vPts1Tmp.emplace_back(pKF1->GetKeyPoint(k).pt);
            vPts2Tmp.emplace_back(pKF2->GetKeyPoint(vMatches[k]).pt);
            vMatchIdx[k] = idx;
            idx++;
        }
    }
    std::cout << "[Matcher::CollectBowMatches] bad parallax_num=" << bad_parallax_num << std::endl;
    average_parallax = sum_parallax / idx;
    return idx;
}


int Matcher::CollectMatches(const KeyFrame *pKF1, const KeyFrame *pKF2, const cv::Mat& F12,
                            vector<int> &vMatches, vector<int> &vMatchIdx,
                            vector<cv::Point2f> &vPts1, vector<cv::Point2f> &vPts2) {
    int idx = 0;
    vMatchIdx.resize(pKF1->N, -1);
    for(int k = 0; k < pKF1->N; k++){
        if(vMatches[k] != -1){
            cv::KeyPoint kp1 = pKF1->GetKeyPointUn(k);
            cv::KeyPoint kp2 = pKF2->GetKeyPointUn(vMatches[k]);
            if(CheckDistEpipolarLine(kp1, kp2, F12)){
                vMatches[k] = -1;
                continue;
            }
            vPts1.emplace_back(kp1.pt);
            vPts2.emplace_back(kp2.pt);
            vMatchIdx[k] = idx;
            idx++;
        }
    }
    return idx;
}

int Matcher::SearchGrid(const cv::Point2f &pt2d, const cv::Mat &description, KeyFrame* pKF2,
                         float radius, int &matchedId, int minLevel, int maxLevel) {

    // 找到以这个点为圆心，radius为半径的圆上的所有像素点所在的grid位置，对这些cell中的点遍历搜索。
    int nMinCellX = std::max(0, (int) std::floor(pt2d.x - radius) / pKF2->mCellSize);
    int nMinCellY = std::max(0, (int) std::floor(pt2d.y - radius) / pKF2->mCellSize);
    int nMaxCellX = std::min(pKF2->mGridCols - 1, (int) std::ceil(pt2d.x + radius) / pKF2->mCellSize);
    int nMaxCellY = std::min(pKF2->mGridRows - 1, (int) std::ceil(pt2d.y + radius) / pKF2->mCellSize);

    int bestDist = INT_MAX;
    int bestId = -1;
    for (int ci = nMinCellY; ci <= nMaxCellY; ci++) {
        for (int cj = nMinCellX; cj <= nMaxCellX; cj++) {
            std::vector<std::size_t> candidatePtsIdx = pKF2->mGrid[ci][cj];
            for (auto j: candidatePtsIdx) {
                cv::KeyPoint kp = pKF2->GetKeyPoint(j);
                if(kp.octave < minLevel || kp.octave > maxLevel) {
                    continue;
                }
                if(fabs(kp.pt.x - pt2d.x) > radius || fabs(kp.pt.y - pt2d.y) > radius) {
                    continue;
                }
                cv::Mat curDesp = pKF2->GetDescription(j);
                int dist = DescriptorDistance(description, curDesp);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestId = j;
                }
            }
        }
    }
//    if (bestDist < 50) {
//        matchedId = bestId;
//        return true;
//    }
//    return false;
    matchedId = bestId;
    return bestDist;
}

int Matcher::SearchGrid(const cv::Point2f &pt2d, const cv::Mat &description, Frame& frame,
                         float radius, int &matchedId, int minLevel, int maxLevel) {

    // 找到以这个点为圆心，radius为半径的圆上的所有像素点所在的grid位置，对这些cell中的点遍历搜索。
    int nMinCellX = std::max(0, (int) std::floor(pt2d.x - radius) / frame.mCellSize);
    int nMinCellY = std::max(0, (int) std::floor(pt2d.y - radius) / frame.mCellSize);
    int nMaxCellX = std::min(frame.mGridCols - 1, (int) std::ceil(pt2d.x + radius) / frame.mCellSize);
    int nMaxCellY = std::min(frame.mGridRows - 1, (int) std::ceil(pt2d.y + radius) / frame.mCellSize);

    int bestDist = INT_MAX;
    int bestId = -1;
    for (int ci = nMinCellY; ci <= nMaxCellY; ci++) {
        for (int cj = nMinCellX; cj <= nMaxCellX; cj++) {
            std::vector<std::size_t> candidatePtsIdx = frame.mGrid[ci][cj];
            for (auto j: candidatePtsIdx) {
                cv::KeyPoint kpUn = frame.mvKeyPointsUn[j];
                if(kpUn.octave < minLevel || kpUn.octave > maxLevel) {
                    continue;
                }
                if(fabs(kpUn.pt.x - pt2d.x) > radius || fabs(kpUn.pt.y - pt2d.y) > radius) {
                    continue;
                }
                cv::Mat curDesp = frame.mDescriptions.row(j);
                int dist = DescriptorDistance(description, curDesp);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestId = j;
                }
            }
        }
    }
//    if (bestDist < 50) {
//        matchedId = bestId;
//        return true;
//    }
//    return false;
    matchedId = bestId;
    return bestDist;
}

std::vector<int> Matcher::SearchByOpticalFlow(KeyFrame* pKF1, KeyFrame* pKF2,
                                              const cv::Mat& img1, const cv::Mat& img2) {
    cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
    std::vector<uchar> status;
    std::vector<float> err;
    std::vector<cv::Point2f> ptsLK;
    // 用KLT对第0层的关键点进行跟踪, ptsLK是跟踪到的点在第二帧中的位置
//    cv::calcOpticalFlowPyrLK(img1, img2, pKF1->GetPointsLevel0(), ptsLK,
//                             status, err, cv::Size(21, 21), 3, criteria);
    cv::calcOpticalFlowPyrLK(img1, img2, pKF1->GetPoints(), ptsLK,
                             status, err, cv::Size(21, 21), 3, criteria);
    std::vector<int> vMatches(pKF1->N, -1);
    std::vector<int> vMatches21(pKF2->N, -1);
    std::vector<int> vMatchDist21(pKF2->N, 256);
    int nMatches = 0;
    for (int i = 0; i < ptsLK.size(); i++) {
        if (status[i] != 1)
            continue;
        cv::Point2f pt = ptsLK[i];
        cv::Mat description = pKF1->GetDescription(i);
        int level = pKF1->GetKeyPoint(i).octave;
        float radiusTh = 15.0;
        float scaleFactor = pKF2->GetScaleFactors()[level];
        float radius = radiusTh * scaleFactor;
        int minLevel = level - 1;
        int maxLevel = level + 1;
        int matchedId;
        int bestDist = SearchGrid(pt, description, pKF2, radius, matchedId, minLevel, maxLevel);
        if (bestDist < 50) {
            if(vMatches21[matchedId] == -1) {
                vMatches[i] = matchedId;
                nMatches++;
                vMatches21[matchedId] = i;
                vMatchDist21[matchedId] = bestDist;
            }
            else{
                if(bestDist < vMatchDist21[matchedId]){
                    vMatches[i] = matchedId;
                    vMatches[vMatches21[matchedId]] = -1;
                    vMatchDist21[matchedId] = bestDist;
                }
            }
        }
    }
    std::cout << "[SearchByOpticalFlow] KL tracking num: " << nMatches << std::endl;
    return vMatches;
}

bool Matcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2,
                                      const cv::Mat &F12) {
    float a = kp1.pt.x * F12.at<float>(0, 0) + kp1.pt.y * F12.at<float>(0, 1) +
              F12.at<float>(0, 2);
    float b = kp1.pt.x * F12.at<float>(1, 0) + kp1.pt.y * F12.at<float>(1, 1) +
              F12.at<float>(1, 2);
    float c = kp1.pt.x * F12.at<float>(2, 0) + kp1.pt.y * F12.at<float>(2, 1) +
              F12.at<float>(2, 2);

    float num = a * kp2.pt.x + b * kp2.pt.y + c;
    float den = a * a + b * b;
    if (den == 0)
        return false;
    float distSqr = num * num / den;
//        std::cout << "[Estimator::CheckDistEpipolarLine] den=" << den << std::endl;
//        std::cout << "[Estimator::CheckDistEpipolarLine] distSqr=" << distSqr<< std::endl;
    return distSqr < 3.84;
}


}