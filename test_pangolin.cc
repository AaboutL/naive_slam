/*
 * @Author: hanfuyong
 * @Date: 2022-08-01 12:23:59
 * @LastEditors: hanfuyong
 * @LastEditTime: 2022-08-01 12:54:05
 * @FilePath: /naive_slam/test_pangolin.cc
 * @Description: 仅用于个人学习
 * 
 * Copyright (c) 2022 by hanfuyong, All Rights Reserved. 
 */
#include <iostream>
#include <pangolin/pangolin.h>

int main(){
    pangolin::CreateWindowAndBind("main", 640, 480);
    pangolin::DataLog log;

    std::vector<std::string> labels;
    labels.push_back(std::string("sin"));
    labels.push_back(std::string("cos"));
    labels.push_back(std::string("sin+cos"));
    log.SetLabels(labels);

    const float tinc = 0.01f;

    pangolin::Plotter plotter(&log, 0.0f, 4.0f*(float)M_PI/tinc, -4.0f, 4.0f, (float)M_PI/(4.0f*tinc), 0.5f);
    plotter.SetBounds(0.0, 1.0, 0.0, 1.0);
    plotter.Track("$i");

    plotter.AddMarker(pangolin::Marker::Vertical, 50*M_PI, pangolin::Marker::LessThan, pangolin::Colour::Blue().WithAlpha(0.2f));
    plotter.AddMarker(pangolin::Marker::Horizontal, 3, pangolin::Marker::GreaterThan, pangolin::Colour::Red().WithAlpha(0.2f));
    plotter.AddMarker(pangolin::Marker::Horizontal, 3, pangolin::Marker::Equal, pangolin::Colour::Green().WithAlpha(0.2f));
    pangolin::DisplayBase().AddDisplay(plotter);

    float t = 0;

    while(!pangolin::ShouldQuit()){
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        log.Log(sin(t), cos(t), sin(t) + cos(t));
        t += tinc;
        pangolin::FinishFrame();
    }

    return 0;
}