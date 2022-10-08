//
// Created by hanfuyong on 2022/9/6.
//

#include "tools.h"

namespace Naive_SLAM{

    cv::Point2f projectPoint(const cv::Mat &pt3d, const cv::Mat& mK) {
        float x_norm = pt3d.at<float>(0) / pt3d.at<float>(2);
        float y_norm = pt3d.at<float>(1) / pt3d.at<float>(2);
        float x_un = x_norm * mK.at<float>(0, 0) + mK.at<float>(0, 2);
        float y_un = y_norm * mK.at<float>(1, 1) + mK.at<float>(1, 2);
        return {x_un, y_un};
    }

    void DrawMatches(const std::string& winName, const cv::Mat& img1, const cv::Mat& img2,
                     const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2,
                     const std::vector<cv::Point2f>& img1_points, const std::vector<cv::Point2f>& img2_points,
                     const cv::Mat& mK, const cv::Mat& distCoff){
        int num = points1.size();
        int w = img1.size().width;
        int h = img1.size().height;

        cv::Mat imgShow(h, w * 2, CV_8UC3, cv::Scalar::all(0));
        cv::Mat tmp, tmpUn;
        if(mK.empty()){
            cv::cvtColor(img1, tmp, cv::COLOR_GRAY2BGR);
            tmp.copyTo(imgShow(cv::Rect(0, 0, w, h)));
            cv::cvtColor(img2, tmp, cv::COLOR_GRAY2BGR);
            tmp.copyTo(imgShow(cv::Rect(w, 0, w, h)));
        }
        else{
            cv::cvtColor(img1, tmp, cv::COLOR_GRAY2BGR);
            cv::undistort(tmp, tmpUn, mK, distCoff, mK);
            tmpUn.copyTo(imgShow(cv::Rect(0, 0, w, h)));

            cv::cvtColor(img2, tmp, cv::COLOR_GRAY2BGR);
            cv::undistort(tmp, tmpUn, mK, distCoff, mK);
            tmpUn.copyTo(imgShow(cv::Rect(w, 0, w, h)));
        }

        for (size_t i = 0; i < img1_points.size(); i++) {
            cv::circle(imgShow, img1_points[i], 3, cv::Scalar(0, 0, 255));
        }
        for (size_t i = 0; i < img2_points.size(); i++) {
            cv::circle(imgShow, (img2_points[i] + cv::Point2f(w, 0)), 3, cv::Scalar(0, 0, 255));
        }
        for (size_t i = 0; i < points1.size(); i++){
            cv::circle(imgShow, points1[i], 3, cv::Scalar(0, 255, 0));
            cv::circle(imgShow, (points2[i] + cv::Point2f(w, 0)), 3, cv::Scalar(0, 255, 0));
            cv::line(imgShow, points1[i], (points2[i] + cv::Point2f(w, 0)), cv::Scalar(255, 0, 0));
        }
        cv::putText(imgShow, std::to_string(num), cv::Point(20, 20), 1, 1, cv::Scalar(255, 0, 0));
        cv::imshow(winName, imgShow);
        cv::waitKey(30);
    }

    void DrawPoints(const cv::Mat& img, const std::vector<cv::Point2f>& pointsDet,
                    const std::vector<cv::Point2f>& pointsTracked, const std::vector<bool>& bOutlier,
                    const cv::Mat& mK, const cv::Mat& distCoff){
        cv::Mat imgShow, imgUn;
        cv::undistort(img, imgUn, mK, distCoff);
        cv::cvtColor(imgUn, imgShow, cv::COLOR_GRAY2BGR);

        // 画出所有检测的点
        for(int i = 0; i < pointsDet.size(); i++) {
            cv::circle(imgShow, pointsDet[i], 5, cv::Scalar(255, 255, 0));
        }

        // 画出投影匹配上的点
        int numInliers = 0;
        int n = pointsTracked.size();
        for(int i = 0; i < n; i++){
            if(!bOutlier[i]){
//            cv::circle(imgShow, pointsTracked[i], 3, cv::Scalar(0, 255, 0));
                cv::rectangle(imgShow, pointsTracked[i] - cv::Point2f(1, 1), pointsTracked[i] + cv::Point2f(1, 1),
                              cv::Scalar(0, 255, 0));
                numInliers++;
            }
            else{
                cv::circle(imgShow, pointsTracked[i], 3, cv::Scalar(0, 0, 255));
            }
        }
        cv::putText(imgShow, "Total tracked points: " + std::to_string(n), cv::Point(20, 35), 1, 1, cv::Scalar(255, 0, 0));
        cv::putText(imgShow, "Inlier points: " + std::to_string(numInliers), cv::Point(20, 20), 1, 1, cv::Scalar(255, 0, 0));
        cv::imshow("Track", imgShow);
        cv::waitKey(30);
    }

    void DrawPoints(const Frame& frame, const std::vector<MapPoint*>& vpMapPoints){
        cv::Mat imgUn, imgShow;
        cv::undistort(frame.mImg, imgUn, frame.mK, frame.mDistCoef);
        cv::cvtColor(imgUn, imgShow, cv::COLOR_GRAY2BGR);

        cv::Mat Rcw = frame.GetRotation();
        cv::Mat tcw = frame.GetTranslation();

        int numInliers = 0;
        for(int i = 0; i < frame.N; i++) {
            MapPoint* pMP = vpMapPoints[i];
            if(!pMP)
                cv::circle(imgShow, frame.mvPointsUn[i], 5, cv::Scalar(255, 0, 0));
            else{
                cv::Mat pt3dcam = Rcw * pMP->GetWorldPos() + tcw;
                cv::Point2f ptProj = projectPoint(pt3dcam, frame.mK);
                cv::rectangle(imgShow, frame.mvPointsUn[i] - cv::Point2f(2, 2),
                              frame.mvPointsUn[i] + cv::Point2f(2, 2), cv::Scalar(0, 255, 0));
                cv::circle(imgShow, ptProj, 3, cv::Scalar(0, 255, 0));
                numInliers++;
            }
        }

        cv::putText(imgShow, "inlier points: " + std::to_string(numInliers), cv::Point(20, 45), 1, 1, cv::Scalar(0, 0, 255));

        cv::imshow("Track", imgShow);
        cv::waitKey(30);
    }

    void DrawPoints(const cv::Mat& img, const std::vector<cv::Point2f>& vPts,
                    const std::vector<MapPoint*>& vMPs,
                    const cv::Mat& mK, const cv::Mat& distCoff,
                    const cv::Mat& Rcw, const cv::Mat& tcw, const std::string& winName, int s) {
        cv::Mat imgShow, imgUn;
        cv::undistort(img, imgUn, mK, distCoff, mK);
        cv::cvtColor(imgUn, imgShow, cv::COLOR_GRAY2BGR);

        // 画出所有检测的点
        for (int i = 0; i < vPts.size(); i++) {
            cv::circle(imgShow, vPts[i], 5, cv::Scalar(255, 0, 0));
        }

        // 画出投影匹配上的点
        int mpsNum = 0;
        for (int i = 0; i < vMPs.size(); i++) {
            if (vMPs[i]) {
                if(!vMPs[i]->IsBad()) {
                    cv::Mat mPt3D = Rcw * vMPs[i]->GetWorldPos() + tcw;
                    if(mPt3D.at<float>(2) <= 0) {
                        continue;
                    }
                    cv::Point2f ptProj = projectPoint(mPt3D, mK);
                    cv::rectangle(imgShow, ptProj - cv::Point2f(s, s), ptProj + cv::Point2f(s, s),
                                  cv::Scalar(0, 255, 0));
                    if(vMPs.size() == vPts.size()){
                        cv::Point2f ptUn = vPts[i];
                        cv::circle(imgShow, ptUn, 3, cv::Scalar(0, 255, 0));
                    }
                    mpsNum++;
                }
            }
        }
        cv::putText(imgShow, "tracked MP num: " + std::to_string(mpsNum), cv::Point(20, 20), 1, 1,
                    cv::Scalar(255, 0, 0));
        cv::imshow(winName, imgShow);
        cv::waitKey(30);
    }

    void DrawPoints(const cv::Mat& img, const KeyFrame* pKF,
                    const cv::Mat& mK, const cv::Mat& distCoff,
                    const std::string& winName, int s, std::vector<float> chi2) {
        cv::Mat imgShow, imgUn;
        cv::undistort(img, imgUn, mK, distCoff, mK);
        cv::cvtColor(imgUn, imgShow, cv::COLOR_GRAY2BGR);

        cv::Mat Rcw = pKF->GetRotation();
        cv::Mat tcw = pKF->GetTranslation();
        std::vector<cv::Point2f> vPts = pKF->GetPointsUn();
        std::vector<MapPoint*> vMPs = pKF->GetMapPoints();
        // 画出所有检测的点
        for (int i = 0; i < vPts.size(); i++) {
            cv::circle(imgShow, vPts[i], 5, cv::Scalar(255, 0, 0));
        }

        // 画出投影匹配上的点
        int mpsNum = 0;
        for (int i = 0; i < vMPs.size(); i++) {
            if (vMPs[i]) {
                if(!vMPs[i]->IsBad()) {
                    cv::Mat mPt3D = Rcw * vMPs[i]->GetWorldPos() + tcw;
                    if(mPt3D.at<float>(2) <= 0) {
//                        std::cout << "z is neg" << std::endl;
                        continue;
                    }
                    cv::Point2f ptProj = projectPoint(mPt3D, mK);
                    cv::rectangle(imgShow, ptProj - cv::Point2f(s, s), ptProj + cv::Point2f(s, s),
                                  cv::Scalar(0, 255, 0));
                    if(vMPs.size() == vPts.size() && !chi2.empty()){
                        cv::Point2f ptUn = vPts[i];
                        cv::circle(imgShow, ptUn, 3, cv::Scalar(0, 255, 0));
                    }
                    mpsNum++;
                }
            }
        }
        cv::putText(imgShow, "mpsNum: " + std::to_string(mpsNum), cv::Point(20, 20), 1, 1,
                    cv::Scalar(255, 0, 0));
        cv::imshow(winName, imgShow);
        cv::waitKey(30);
    }

    void PrintMat(const std::string& msg, const cv::Mat& mat){
        std::cout << msg << std::endl;
        for(int i = 0; i < mat.rows; i++){
            for(int j = 0; j < mat.cols; j++){
                std::cout << mat.at<float>(i, j) << "  ";
            }
            std::cout << std::endl;
        }
    }

    void PrintTime(const std::string& msg, const std::chrono::system_clock::time_point& t){
        std::chrono::system_clock::time_point t1 = std::chrono::system_clock::now();
        double dur = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t).count();
        std::cout << msg << " " << dur << std::endl;
    }



}
