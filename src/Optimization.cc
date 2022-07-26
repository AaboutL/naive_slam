//
// Created by hanfuyong on 2022/8/10.
//

#include "Optimization.h"
#include "Converter.h"

#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/types_six_dof_expmap.h> // 此头文件包含了6自由度pose顶点和各种边的头文件


namespace Naive_SLAM {

    int Optimization::PoseOptimize(const std::vector<cv::KeyPoint> &vKPsUn,
                                   std::vector<MapPoint *> &vMPs,
                                   const std::vector<float>& vInvLevelSigma2,
                                   const cv::Mat &matK, cv::Mat &Tcw, std::vector<bool> &outlier,
                                   std::vector<float> &chi2s) {
        // 定义g2o优化器，可看作总管
        g2o::SparseOptimizer optimizer;
//    optimizer.setVerbose(true);

        // 定义线性求解器，是BlockSolver类中的成员。
        // LinearSolverType是在BlockSolver类中对LinearSolver类的一个类型定义
        std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver =
                g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();

        // 定义BlockSolver求解器，并把线性求解器传入
        // BlockSolver有两个成员，一个是负责计算Hx=-b的LinearSolver；一个是计算jacobian和hessian矩阵的SparseBlockMatrix。
        std::unique_ptr<g2o::BlockSolver_6_3> solver_ptr = g2o::make_unique<g2o::BlockSolver_6_3>(
                std::move(linearSolver));

        // 定义迭代优化方法
        // 迭代优化方法类中包含一个BlockSolver求解器
        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(
                std::move(solver_ptr));

        // 设置g2o优化器的迭代优化算法。
        // g2o优化器中包含一个迭代优化方法
        optimizer.setAlgorithm(solver);

        // 添加Pose顶点：待优化的当前帧的位姿Tcw
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::TtoSE3Quat(Tcw));
        vSE3->setId(0);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);

        // 自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值5.991
        const float chi2Mono = 5.991;
        const float deltaMono = sqrt(chi2Mono);

        std::vector<g2o::EdgeSE3ProjectXYZOnlyPose *> monoEdges;
        std::vector<int> monoEdgesIdx;
        int N = static_cast<int>(vMPs.size());
        outlier.resize(N, true);
        // 添加MapPoints顶点：
        for (int i = 0; i < N; i++) {
            MapPoint *pMP = vMPs[i];
            if (pMP) {
                outlier[i] = false;
                // 新建图的边，为一元边，因为需要优化的只有pose一项
                auto *e = new g2o::EdgeSE3ProjectXYZOnlyPose();

                // 给边添加顶点
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));

                // 设置相机内参
                e->fx = matK.at<float>(0, 0);
                e->fy = matK.at<float>(1, 1);
                e->cx = matK.at<float>(0, 2);
                e->cy = matK.at<float>(1, 2);

                // 给边设置观测量
                Eigen::Vector2d obs;
                obs << vKPsUn[i].pt.x, vKPsUn[i].pt.y;
                e->setMeasurement(obs);

                // 设置地图点的空间位置
                e->Xw[0] = pMP->GetWorldPos().at<float>(0);
                e->Xw[1] = pMP->GetWorldPos().at<float>(1);
                e->Xw[2] = pMP->GetWorldPos().at<float>(2);

                float invSigma2 = vInvLevelSigma2[vKPsUn[i].octave];
                e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
                // 设置鲁棒和函数
                auto *rk = new g2o::RobustKernelHuber();
                rk->setDelta(deltaMono);
                e->setRobustKernel(rk);

                // 把边添加到图优化器中
                optimizer.addEdge(e);
                monoEdges.emplace_back(e);
                monoEdgesIdx.emplace_back(i);
            }
        }
        if (monoEdges.size() < 3)
            return 0;

        // 进行4次优化，每次优化迭代10次
        // 每次优化之后，把所有点对重新分成inlier和outlier，下次迭代只用inlier的数据进行优化
        chi2s.resize(outlier.size(), 0);
        int nBad = 0;
        for (int k = 0; k < 4; k++) {
            nBad = 0;
            vSE3->setEstimate(Converter::TtoSE3Quat(Tcw));
            optimizer.initializeOptimization(0);
            optimizer.optimize(10);

            // 每次优化结束后，分析每个边的误差
            for (size_t j = 0; j < monoEdges.size(); j++) {
                auto *e = monoEdges[j];
                int idx = monoEdgesIdx[j];
                if (outlier[idx]) { // 上一次标记为outlier的点，因为没有参与本次优化，不知道当前的误差，所以需要单独计算
                    e->computeError();
                }
                float chi2 = e->chi2();
                chi2s[idx] = chi2;
                if (chi2 < chi2Mono) {
                    outlier[idx] = false;
                    e->setLevel(0); // level设置为0，表示参与优化
                } else {
                    outlier[idx] = true;
                    e->setLevel(1); // level设置为1，表示不参与优化
                    nBad++;
                }
            }
            if (optimizer.edges().size() < 10)
                break;
        }

        for(int i = 0; i < N; i++){
            if(outlier[i]){
                vMPs[i] = nullptr;
            }
        }

        g2o::VertexSE3Expmap *vSE3_opt = dynamic_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
        g2o::SE3Quat SE3Quat_opt = vSE3_opt->estimate();
        Tcw = Converter::SE3toT(SE3Quat_opt).clone();
        int nInlierNum = monoEdges.size() - nBad;
        std::cout << "[Optimization::PoseOptimize] g2o inliers num=" << nInlierNum << std::endl;
        return nInlierNum;
    }

    void Optimization::SlidingWindowBA(vector<KeyFrame *> &vpKFs, const cv::Mat &matK) {
        g2o::SparseOptimizer optimizer;
//    optimizer.setVerbose(true);
        std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver =
                std::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
        std::unique_ptr<g2o::BlockSolver_6_3> solver_ptr =
                std::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver));
        auto *solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));
        optimizer.setAlgorithm(solver);

        float thHuberMono = sqrt(5.991);

        int vertexIdx = 0;
        // 遍历滑窗内的关键帧，添加g2o的pose顶点
        for (int i = 0; i < vpKFs.size(); i++) {
            KeyFrame *pKF = vpKFs[i];
            auto *vSE3 = new g2o::VertexSE3Expmap();
            vSE3->setEstimate(Converter::TtoSE3Quat(pKF->GetTcw()));
            vSE3->setId(vertexIdx);
            if (i == 0) {
                vSE3->setFixed(true);
            }
            optimizer.addVertex(vSE3);
            vertexIdx++;
        }

        // 遍历滑窗中关键帧对应的地图点
        std::map<MapPoint *, int> mMPWithIdx;
        std::vector<g2o::EdgeSE3ProjectXYZ *> vpEdges;
        std::vector<KeyFrame *> vpEdgeKFs;
        std::vector<MapPoint *> vpEdgeMPs;
        for (int i = 0; i < vpKFs.size(); i++) {
            KeyFrame *pKF = vpKFs[i];
            std::vector<MapPoint *> vpMPs = pKF->GetMapPoints();
            for (int j = 0; j < vpMPs.size(); j++) {
                MapPoint *pMP = vpMPs[j];
                if (pMP) {
                    auto *e = new g2o::EdgeSE3ProjectXYZ();
                    if (mMPWithIdx.find(pMP) == mMPWithIdx.end()) {
                        auto *vPoint = new g2o::VertexPointXYZ();
                        vPoint->setEstimate(Converter::cvMatToEigenVector(pMP->GetWorldPos()));
                        vPoint->setId(vertexIdx);
                        vPoint->setMarginalized(true);
                        optimizer.addVertex(vPoint);
                        e->setVertex(0,
                                     dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(
                                             vertexIdx)));
                        mMPWithIdx[pMP] = vertexIdx;
                        vertexIdx++;
                    } else {
                        e->setVertex(0,
                                     dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(
                                             mMPWithIdx[pMP])));
                    }

                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(
                            i)));
                    Eigen::Matrix<double, 2, 1> obs;
                    cv::KeyPoint kpUn = pKF->GetKeyPointUn(j);
                    obs << kpUn.pt.x, kpUn.pt.y;
                    e->setMeasurement(obs);
                    float invSigma2 = pKF->GetInvLevelSigma2()[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    auto *rk = new g2o::RobustKernelHuber();
                    rk->setDelta(thHuberMono);
                    e->setRobustKernel(rk);
                    e->fx = matK.at<float>(0, 0);
                    e->fy = matK.at<float>(1, 1);
                    e->cx = matK.at<float>(0, 2);
                    e->cy = matK.at<float>(1, 2);

                    optimizer.addEdge(e);

                    vpEdges.emplace_back(e);
                    vpEdgeKFs.emplace_back(pKF);
                    vpEdgeMPs.emplace_back(pMP);
                }
            }
        }

        // 第一遍优化
        std::cout << "[SlidingWindowBA] first optimizing start..." << std::endl;
        optimizer.initializeOptimization();
        optimizer.optimize(5);
        std::cout << "[SlidingWindowBA] first optimize done" << std::endl;

        // 第二遍优化
        for (size_t i = 0; i < vpEdges.size(); i++) {
            g2o::EdgeSE3ProjectXYZ *e = vpEdges[i];
            MapPoint *pMP = vpEdgeMPs[i];
            if(pMP->IsBad())
                continue;
            if (e->chi2() > 5.991 || !e->isDepthPositive()) {
                e->setLevel(1);
            }
            e->setRobustKernel(nullptr);
        }
        std::cout << "[SlidingWindowBA] second optimizing start..." << std::endl;
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        std::cout << "[SlidingWindowBA] second optimize done" << std::endl;

        int outlierNum1 = 0, outlierNum2 = 0;
        for (size_t i = 0; i < vpEdges.size(); i++) {
            g2o::EdgeSE3ProjectXYZ *e = vpEdges[i];
            if (e->chi2() > 5.991 || !e->isDepthPositive()) {
                if(e->chi2() > 5.991) outlierNum1++;
                if(!e->isDepthPositive()) outlierNum2++;
                KeyFrame *pKF = vpEdgeKFs[i];
                MapPoint *pMP = vpEdgeMPs[i];
                pKF->EraseMapPoint(pMP);
                pMP->EraseObservation(pKF);
            }
        }
        std::cout << "[Optimization::SlidingWindowBA] edge num=" << vpEdges.size()
                  << "  chi2 outlier num=" << outlierNum1
                  << "  depth outlier num=" << outlierNum2 << std::endl;

        for (int i = 0; i < vpKFs.size(); i++) {
            KeyFrame *pKF = vpKFs[i];
            auto *vSE3 = dynamic_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(i));
            g2o::SE3Quat SE3Quat = vSE3->estimate();
            pKF->SetT(Converter::SE3toT(SE3Quat).clone());
        }
        for (auto &it: mMPWithIdx) {
            MapPoint *pMP = it.first;
            auto *vPoint = dynamic_cast<g2o::VertexPointXYZ *>(optimizer.vertex(it.second));
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        }
    }

    bool Optimization::SolvePnP(const std::vector<cv::Point2f>& vPtsUn,
                                vector<MapPoint *> &vpMapPoints, const cv::Mat& K, cv::Mat& Tcw) {
        std::vector<cv::Point3f> vPts3D;
        std::vector<cv::Point2f> vPts2D;
        std::vector<int> vMatchIdx;
        for (int id = 0; id < vPtsUn.size(); id++) {
            MapPoint* pMP = vpMapPoints[id];
            if (!pMP || pMP->IsBad())
                continue;
            vMatchIdx.emplace_back(id);
            vPts3D.emplace_back(pMP->GetWorldPos());
            vPts2D.emplace_back(vPtsUn[id]);
        }
        std::cout << "[Optimization::SolvePnP] 3D 2D matched for PnP num=" << vPts3D.size() << std::endl;
        if(vPts3D.empty())
            return false;
        cv::Mat rcw, tcw, Rcw, inliers;
        cv::solvePnPRansac(vPts3D, vPts2D, K, cv::Mat::zeros(4, 1, CV_32F),
                           rcw, tcw, false, 100, 4, 0.99, inliers, cv::SOLVEPNP_EPNP);
        std::cout << "[Optimization::SolvePnP] pnp inliers num=" << inliers.total() << std::endl;
        if(inliers.total() < 10)
            return false;
        cv::Rodrigues(rcw, Rcw);
        Rcw.convertTo(Rcw, CV_32F);
        tcw.convertTo(tcw, CV_32F);
        Tcw = cv::Mat::eye(4, 4, CV_32F);
        Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
        tcw.copyTo(Tcw.rowRange(0, 3).col(3));

        std::vector<int> vNoMatchIdx = vMatchIdx;
        for(int k = 0; k < inliers.total(); k++){
            vNoMatchIdx[inliers.at<int>(k)] = -1;
        }
        for(int k : vNoMatchIdx){
            if(k != -1){
                vpMapPoints[k] = nullptr;
            }
        }
        return true;
    }

}