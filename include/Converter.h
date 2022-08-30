//
// Created by hanfuyong on 2022/8/12.
//

#ifndef NAIVE_SLAM_CONVERTER_H
#define NAIVE_SLAM_CONVERTER_H

#include <vector>
#include <Eigen/Dense>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <opencv2/opencv.hpp>

namespace Naive_SLAM{

class Converter{
public:
    static g2o::SE3Quat TtoSE3Quat(const cv::Mat& T){
        Eigen::Matrix3d R;
        R << T.at<float>(0, 0), T.at<float>(0, 1), T.at<float>(0, 2),
             T.at<float>(1, 0), T.at<float>(1, 1), T.at<float>(1, 2),
             T.at<float>(2, 0), T.at<float>(2, 1), T.at<float>(2, 2);
        Eigen::Vector3d t;
        t << T.at<float>(0, 3), T.at<float>(1, 3), T.at<float>(2, 3);

        return g2o::SE3Quat(R, t);
    }

    static cv::Mat SE3toT(const g2o::SE3Quat& SE3Quat){
        cv::Mat T = cv::Mat::zeros(4, 4, CV_32F);
        Eigen::Matrix<double,4,4> eigMat = SE3Quat.to_homogeneous_matrix();
        for (int i = 0; i < 4; i++){
            for (int j = 0; j < 4; j++){
                T.at<float>(i, j) = eigMat(i, j);
            }
        }
        return T;
    }

    static cv::Mat toCvMat(const Eigen::Matrix<double,3,1> &m){
        cv::Mat mat(3, 1, CV_32F);
        for(int i = 0; i < 3; i++){
            mat.at<float>(i) = m(i);
        }
        return mat.clone();
    }

    static std::vector<cv::Mat> DescriptionMatToVector(const cv::Mat& description){
        std::vector<cv::Mat> vDesc;
        vDesc.reserve(description.rows);
        for(int j = 0; j < description.rows; j++){
            vDesc.emplace_back(description.row(j));
        }
        return vDesc;
    }

    static Eigen::Vector3d cvMatToEigenVector(const cv::Mat& m){
        Eigen::Vector3d eVec;
        eVec << m.at<float>(0), m.at<float>(1), m.at<float>(2);
        return eVec;
    }

};

}

#endif //NAIVE_SLAM_CONVERTER_H
