/*
 * @Author: hanfuyong
 * @Date: 2022-08-01 15:01:08
 * @LastEditors: hanfuyong
 * @LastEditTime: 2022-08-01 17:22:36
 * @FilePath: /naive_slam/test_pangolin_euroc.cc
 * @Description: 仅用于个人学习
 * 
 * Copyright (c) 2022 by hanfuyong, All Rights Reserved. 
 */
#include <iostream>
#include <pangolin/pangolin.h>
#include <eigen3/Eigen/Eigen>
#include <fstream>
#include <sstream>


int main(){
    std::ifstream fin;
    fin.open("/home/aal/dataset/slam/EuRoC/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv", std::ios::in);
    if(!fin.is_open()){
        std::cout << "not opened" << std::endl;
    }
    std::string s;
    getline(fin, s);

    pangolin::CreateWindowAndBind("camera_pose", 752*2, 480*2);
    glEnable(GL_DEPTH_TEST);
    pangolin::OpenGlRenderState s_cam_ = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(752*2, 480*2, 420, 420, 320, 240, 0.1, 1000),
        pangolin::ModelViewLookAt(5, -3, 5, 0, 0, 0, pangolin::AxisZ)
    );

    pangolin::View& d_cam_ = pangolin::CreateDisplay()
        .SetBounds(0., 1., 0., 1., -752/480.)
        .SetHandler(new pangolin::Handler3D(s_cam_));
    
    std::vector<Eigen::Vector3d> traj;
    ulong time_stamp(0);
    double px(0.), py(0.), pz(0.);
    double qw(0.), qx(0.), qy(0.), qz(0.);
    double vx(0.), vy(0.), vz(0.);
    double bwx(0.), bwy(0.), bwz(0.), bax(0.), bay(0.), baz(0.);
    while(!fin.eof()){
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam_.Activate(s_cam_);
        std::string line;
        getline(fin, line);
        std::stringstream ss(line);
        std::vector<float> v;
        std::string s_tmp;
        
        getline(ss, s_tmp, ',');
        time_stamp = std::stoul(s_tmp);
        while(getline(ss, s_tmp, ',')){
            v.emplace_back(std::stod(s_tmp));
        }
        px = v[0], py=v[1], pz=v[2];
        qw = v[3], qx=v[4], qy=v[5], qz=v[6];
        vx = v[7], vy = v[8], vz = v[9];
        bwx = v[10], bwy = v[11], bwz = v[12];
        bax = v[13], bay = v[14], baz = v[15];

        Eigen::Quaterniond quat(qw, qx, qy, qz);
        Eigen::Vector3d pos(px, py, pz);
        traj.emplace_back(pos);

        glLineWidth(3);
        glBegin(GL_LINES);
        glColor3f ( 1.0f,0.f,0.f );
	    glVertex3f( 0,0,0 );
	    glVertex3f( 1,0,0 );
	    glColor3f( 0.f,1.0f,0.f);
	    glVertex3f( 0,0,0 );
	    glVertex3f( 0,1,0 );
	    glColor3f( 0.f,0.f,1.f);
	    glVertex3f( 0,0,0 );
	    glVertex3f( 0,0,1 );
	    glEnd();

        Eigen::Matrix3d R = quat.toRotationMatrix();
        
        glPushMatrix();
        std::vector<GLdouble> Twc = {R(0, 0), R(1,0), R(2, 0), 0.,
                                R(0, 1), R(1, 1), R(2, 1), 0.,
                                R(0, 2), R(1, 2), R(2, 2), 0.,
                                pos.x(), pos.y(), pos.z(), 1.};
        glMultMatrixd(Twc.data());
        const float w = 0.2;
        const float h = w * 0.75;
        const float z = w * 0.6;

        glLineWidth(2); 
        glBegin(GL_LINES);
        glColor3f(0.0f,1.0f,1.0f);
	    glVertex3f(0,0,0);		glVertex3f(w,h,z);
	    glVertex3f(0,0,0);		glVertex3f(w,-h,z);
	    glVertex3f(0,0,0);		glVertex3f(-w,-h,z);
	    glVertex3f(0,0,0);		glVertex3f(-w,h,z);
	    glVertex3f(w,h,z);		glVertex3f(w,-h,z);
	    glVertex3f(-w,h,z);		glVertex3f(-w,-h,z);
	    glVertex3f(-w,h,z);		glVertex3f(w,h,z);
	    glVertex3f(-w,-h,z);    glVertex3f(w,-h,z);
	    glEnd();
        glPopMatrix();

        // -------- 绘制相机轨迹 --------//
        glLineWidth(2);
        glBegin(GL_LINES);
        glColor3f(0.f, 1.f, 0.f);
        for(size_t i=0; i<traj.size() - 1; i++){
            glVertex3d(traj[i].x(), traj[i].y(), traj[i].z());
            glVertex3d(traj[i+1].x(), traj[i+1].y(), traj[i+1].z());
        }
        glEnd();
    
        pangolin::FinishFrame();
        
        if(pangolin::ShouldQuit())
            break;
    }
    return 0;
}