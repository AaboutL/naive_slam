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


namespace Naive_SLAM{

int Optimization::PoseOptimize(const std::vector<cv::Point2f> &ptsUn, const std::vector<MapPoint *> &mapPoints,
                                const cv::Mat &matK, cv::Mat& Tcw, std::vector<bool>& outlier) {
    // 定义g2o优化器，可看作总管
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(true);

    // 定义线性求解器，是BlockSolver类中的成员。
    // LinearSolverType是在BlockSolver类中对LinearSolver类的一个类型定义
    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver =
            g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();

    // 定义BlockSolver求解器，并把线性求解器传入
    // BlockSolver有两个成员，一个是负责计算Hx=-b的LinearSolver；一个是计算jacobian和hessian矩阵的SparseBlockMatrix。
    std::unique_ptr<g2o::BlockSolver_6_3> solver_ptr = g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver));

    // 定义迭代优化方法
    // 迭代优化方法类中包含一个BlockSolver求解器
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));

    // 设置g2o优化器的迭代优化算法。
    // g2o优化器中包含一个迭代优化方法
    optimizer.setAlgorithm(solver);

    // 添加Pose顶点：待优化的当前帧的位姿Tcw
    g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::TtoSE3Quat(Tcw));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // 自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值5.991
    const float chi2Mono = 5.991;
    const float deltaMono = sqrt(chi2Mono);

    std::vector<g2o::EdgeSE3ProjectXYZOnlyPose*> monoEdges;
    std::vector<int> monoEdgesIdx;
    int n = mapPoints.size();
    outlier.resize(n, true);
    // 添加MapPoints顶点：
    for(int i = 0; i < n; i++){
        MapPoint* pMP = mapPoints[i];
        if(pMP){
            outlier[i] = false;
            // 新建图的边，为一元边，因为需要优化的只有pose一项
            g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

            // 给边添加顶点
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));

            // 设置相机内参
            e->fx = matK.at<float>(0, 0);
            e->fy = matK.at<float>(1, 1);
            e->cx = matK.at<float>(0, 2);
            e->cy = matK.at<float>(1, 2);

            // 给边设置观测量
            Eigen::Vector2d obs;
            obs << ptsUn[i].x, ptsUn[i].y;
            e->setMeasurement(obs);

            // 设置地图点的空间位置
            e->Xw[0] = pMP->GetWorldPos().x;
            e->Xw[1] = pMP->GetWorldPos().y;
            e->Xw[2] = pMP->GetWorldPos().z;

            e->setInformation(Eigen::Matrix2d::Identity());
            // 设置鲁棒和函数
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
            rk->setDelta(deltaMono);
            e->setRobustKernel(rk);

            // 把边添加到图优化器中
            optimizer.addEdge(e);
            monoEdges.emplace_back(e);
            monoEdgesIdx.emplace_back(i);
        }
    }
    if(monoEdges.size() < 3)
        return 0;

    // 进行4次优化，每次优化迭代10次
    // 每次优化之后，把所有点对重新分成inlier和outlier，下次迭代只用inlier的数据进行优化
    int nBad = 0;
    for(int k = 0; k < 4; k++){
        vSE3->setEstimate(Converter::TtoSE3Quat(Tcw));
        optimizer.initializeOptimization(0);
        optimizer.optimize(10);

        nBad = 0;
        // 每次优化结束后，分析每个边的误差
        for(size_t j = 0; j < monoEdges.size(); j++){
            auto* e = monoEdges[j];
            int idx = monoEdgesIdx[j];
            if(outlier[idx]){ // 上一次标记为outlier的点，因为没有参与本次优化，不知道当前的误差，所以需要单独计算
                e->computeError();
            }
            float chi2 = e->chi2();
            if(chi2 < chi2Mono){
                outlier[idx] = false;
                e->setLevel(0); // level设置为0，表示参与优化
            }
            else{
                outlier[idx] = true;
                e->setLevel(1); // level设置为1，表示不参与优化
                nBad ++;
            }
        }
        if(optimizer.edges().size()<10)
            break;
    }

    g2o::VertexSE3Expmap* vSE3_opt = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3Quat_opt = vSE3_opt->estimate();
    Tcw = Converter::SE3toT(SE3Quat_opt).clone();
    return monoEdges.size() - nBad;
}

}