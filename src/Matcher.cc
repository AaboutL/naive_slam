//
// Created by hanfuyong on 2022/8/30.
//

#include "Matcher.h"

namespace Naive_SLAM{

int Matcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

int Matcher::SearchByBow(Naive_SLAM::KeyFrame *pKF1, Naive_SLAM::KeyFrame *pKF2, std::vector<int> &matches) {
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
    while(f1it != f1end && f2it != f2end){
        if(f1it->first == f2it->first){
            for(size_t i1 = 0; i1 < f1it->second.size(); i1++){
                const int idx1 = static_cast<int>(f1it->second[i1]);
                MapPoint* pMP1 = pKF1->GetMapPoint(idx1);
                if(pMP1)
                    continue;

                cv::KeyPoint kp1 = pKF1->GetKeyPointUn(idx1);
                cv::Mat description1 = pKF1->GetDescription(idx1);

                int bestDist = 50;
                int bestIdx = -1;
                for (size_t i2 = 0; i2 < f2it->second.size(); i2++){
                    const int idx2 = static_cast<int>(f2it->second[i2]);
                    MapPoint* pMP2 = pKF2->GetMapPoint(idx2);
                    if (pMP2 && !vbMatched2[idx2])
                        continue;

                    cv::KeyPoint kp2 = pKF2->GetKeyPointUn(idx2);
                    cv::Mat description2 = pKF2->GetDescription(idx2);
                    int dist = DescriptorDistance(description1, description2);
                    if(dist < bestDist){
                        bestDist = dist;
                        bestIdx = idx2;
                    }
                }
                if(bestIdx >= 0){
                    matches[idx1] = bestIdx;
                    vbMatched2[bestIdx] = true;
                    matchNum++;
                }
            }
            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first) {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }
    return matchNum;
}

}