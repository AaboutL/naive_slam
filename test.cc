#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <chrono>
#include "ORBextractor.h"

void draw(const cv::Mat& img, const std::vector<cv::KeyPoint>& keyPoints, const std::string& win_name="test"){
    cv::Mat img_tmp;
    img.copyTo(img_tmp);
    for (const auto& kp : keyPoints){
        cv::circle(img_tmp, kp.pt, 3, cv::Scalar(255, 255, 255));
    }
    cv::imshow(win_name, img_tmp);
    cv::waitKey(0);
}

void DrawFlow(const cv::Mat& img, const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2, 
    const std::vector<uchar>& status){

    int cnt = 0;
    cv::Mat img_tmp;
    img.copyTo(img_tmp);
    for (size_t i = 0; i < points1.size(); i++){
        cv::circle(img_tmp, points1[i], 3, cv::Scalar(255, 255, 255));
        if (status[i]){
            cnt++;
            cv::line(img_tmp, points1[i], points2[i], cv::Scalar(255, 255, 255));
        }
    }
    // std::cout << cnt << std::endl;
    cv::imshow("klt", img_tmp);
    cv::waitKey(0);
}

void DrawMatches(const cv::Mat& img1, const cv::Mat& img2, const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2){
    int w = img1.size().width;
    int h = img1.size().height;
    cv::Mat imgShow(h, w * 2, CV_8UC3, cv::Scalar::all(0));
    cv::Mat tmp;
    cv::cvtColor(img1, tmp, cv::COLOR_GRAY2BGR);
    tmp.copyTo(imgShow(cv::Rect(0, 0, w, h)));
    cv::cvtColor(img2, tmp, cv::COLOR_GRAY2BGR);
    tmp.copyTo(imgShow(cv::Rect(w, 0, w, h)));
    cv::resize(imgShow, imgShow, imgShow.size() * 2);
    for (size_t i = 0; i < points1.size(); i++){
        cv::circle(imgShow, points1[i] * 2, 3, cv::Scalar(255, 0, 0));
        cv::circle(imgShow, (points2[i] + cv::Point2f(w, 0)) * 2, 3, cv::Scalar(0, 255, 0));
        cv::line(imgShow, points1[i] * 2, (points2[i] + cv::Point2f(w, 0)) * 2, cv::Scalar(255, 0, 0));
    }
    cv::imshow("match", imgShow);
    cv::waitKey(0);
}

int DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
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

std::vector<int> SearchInArea(const std::vector<cv::Point2f>& pts1, const std::vector<uchar>& status, const std::vector<int>& indices, 
    const std::vector<cv::KeyPoint>& kpts, const cv::Mat& desps, const std::vector<cv::KeyPoint>& kpts1, const cv::Mat& desps1,
    const int cellSize, const cv::Size& imgSize){
    
    std::vector<int> matchIdx(indices.size(), -1);
    for (size_t i = 0; i < indices.size(); i++){
        if (!status[i]) 
            continue;
        cv::Point2f pt = pts1[i];
        int minCellX = std::max(0, (int)pt.x - cellSize);
        int maxCellX = std::min(imgSize.width, (int)pt.x + cellSize);
        int minCellY = std::max(0, (int)pt.y - cellSize);
        int maxCellY = std::min(imgSize.height, (int)pt.y + cellSize);
        // std::cout << pt << "  " << minCellX <<  " " << maxCellX << " " << minCellY << " " << maxCellY << std::endl;
        int bestDist = 100000;
        int bestId = -1;
        for (size_t j = 0; j < kpts1.size(); j++){
            if (kpts1[j].octave != 0)
                continue;
            if (kpts1[j].pt.x < minCellX || kpts1[j].pt.x > maxCellX || kpts1[j].pt.y < minCellY || kpts1[j].pt.y > maxCellY)
                continue;
            // std::cout << kpts1[j].pt << std::endl;
            cv::Mat desp = desps.row(indices[i]);
            cv::Mat desp1 = desps1.row(j);
            int dist = DescriptorDistance(desp, desp1);
            if(dist < bestDist){
                bestDist = dist;
                bestId = j;
            }
        }
        // std::cout << "bestDist: " << bestDist << std::endl;
        if (bestDist < 40){
            matchIdx[i] = bestId;
        }
    }
    return matchIdx;
}

int main(int argc, char** argv){
    cv::Mat img = cv::imread("/home/aal/dataset/slam/MH_01_easy/mav0/cam0/data/1403636579763555584.png");
    cv::Mat img1 = cv::imread("/home/aal/dataset/slam/MH_01_easy/mav0/cam0/data/1403636580013555456.png");
    std::cout << img.size() << std::endl;
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img1, img1, cv::COLOR_BGR2GRAY);

    // cv::Ptr<cv::ORB> orb = cv::ORB::create(1000, 1.2, 8, 31, 0, 2, cv::ORB::FAST_SCORE, 20);
    // std::vector<cv::KeyPoint> keyPoints;
    // cv::Mat descriptions;
    // orb->detect(img, keyPoints);
    // orb->compute(img, keyPoints, descriptions);
    // draw(img, keyPoints, "show_cv");

    // cv::Ptr<cv::ORB> orb1 = cv::ORB::create(1000, 1.2, 8, 31, 0, 2, cv::ORB::FAST_SCORE, 20);
    // std::vector<cv::KeyPoint> keyPoints1;
    // cv::Mat descriptions1;
    // orb->detect(img1, keyPoints1);
    // orb->compute(img1, keyPoints1, descriptions1);
    // draw(img1, keyPoints1, "show_cv1");

    std::chrono::system_clock::time_point t0 = std::chrono::system_clock::now();
    std::vector<cv::KeyPoint> keyPoints;
    cv::Mat descriptions;
    Naive_SLAM::ORBextractor* orb = new Naive_SLAM::ORBextractor(1000, 1.2, 8, 20, 7);
    (*orb)(img, cv::Mat(), keyPoints, descriptions);
    // draw(img, keyPoints, "show_orb");

    std::vector<cv::KeyPoint> keyPoints1;
    cv::Mat descriptions1;
    Naive_SLAM::ORBextractor* orb1 = new Naive_SLAM::ORBextractor(1000, 1.2, 8, 20, 7);
    (*orb1)(img1, cv::Mat(), keyPoints1, descriptions1);
    // draw(img1, keyPoints1, "show_orb");
    std::chrono::system_clock::time_point t1 = std::chrono::system_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
    std::cout << "orb duration: " << duration << std::endl;

    std::vector<cv::Point2f> pts, pts1;
    std::vector<int> indices;
    for (size_t i = 0; i < keyPoints.size(); i++){
        cv::KeyPoint kpt = keyPoints[i];
        if (kpt.octave == 0){
            pts.emplace_back(kpt.pt);
            indices.emplace_back(i);
        }
    }

    std::chrono::system_clock::time_point t2 = std::chrono::system_clock::now();
    cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(img, img1, pts, pts1, status, err, cv::Size(21, 21), 8, criteria);
    std::chrono::system_clock::time_point t3 = std::chrono::system_clock::now();
    double duration1 = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2).count();
    std::cout << "lkt :" << duration1 << std::endl;
    DrawFlow(img, pts, pts1, status);
    std::cout << pts.size() << "  " << pts1.size() << "  " << status.size() << "  " << err.size() << std::endl;

    std::chrono::system_clock::time_point t4 = std::chrono::system_clock::now();
    std::vector<int> matchIdx = SearchInArea(pts1, status, indices, keyPoints, descriptions, keyPoints1, descriptions1, 10, img.size());
    std::chrono::system_clock::time_point t5 = std::chrono::system_clock::now();
    double dur = std::chrono::duration_cast<std::chrono::duration<double>>(t5 - t4).count();
    std::cout  << "search :" << dur << std::endl;
    std::vector<cv::Point2f> ptsMch, pts1Mch;
    for (size_t i = 0; i < matchIdx.size(); i++){
        if (matchIdx[i] == -1)
            continue;
        ptsMch.emplace_back(keyPoints[indices[i]].pt);
        pts1Mch.emplace_back(keyPoints1[matchIdx[i]].pt);
    }
    std::cout << ptsMch.size() << "  " << pts1Mch.size() << std::endl;
    DrawMatches(img, img1, ptsMch, pts1Mch);
    
    return 0;
}