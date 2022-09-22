#include <iostream>
#include <math.h>
#include <chrono>
#include <thread>
#include <opencv2/opencv.hpp>
//#include <pangolin/pangolin.h>
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
//    cv::resize(imgShow, imgShow, imgShow.size() * 2);
    for (size_t i = 0; i < points1.size(); i++){
        cv::circle(imgShow, points1[i], 3, cv::Scalar(255, 0, 0));
        cv::circle(imgShow, (points2[i] + cv::Point2f(w, 0)), 3, cv::Scalar(0, 255, 0));
        cv::line(imgShow, points1[i], (points2[i] + cv::Point2f(w, 0)), cv::Scalar(255, 0, 0));
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
            std::cout << bestDist << " ";
        }
    }
    return matchIdx;
}

void SampleMethod(){
    std::cout << "You typed ctrl-r or pushed reset" << std::endl;
}

static const std::string window_name = "HelloPangolinThreads";


//void setup() {
//    // create a window and bind its context to the main thread
//    // 创建名称为window_name的窗口，视窗的长宽分别为640、480
//    pangolin::CreateWindowAndBind(window_name, 640, 480);
//
//    // enable depth
//    // 启动深度测试，使得pangolin只会绘制朝向镜头的那一面像素点，避免容易混淆的透视关系出现。
//    glEnable(GL_DEPTH_TEST);
//
//    // unset the current context from the main thread
//    pangolin::GetBoundWindow()->RemoveCurrent();
//}
//
//
//void run() {
//    // fetch the context and bind it to this thread
//    pangolin::BindToContext(window_name);
//
//    // we manually need to restore the properties of the context
//    glEnable(GL_DEPTH_TEST);
//
//    // Define Projection and initial ModelView matrix
//    // 创建一个观察相机，用于观察，而不是slam中的相机
//    // ProjectionMatrix是观察相机的内参矩阵，在进行交互操作时，pangolin会自动根据内参矩阵完成对应的透视变换
//    // ModelViewLookAt给出观察相机初始时刻的位置、相机的视点位置（即相机的光轴朝向的点）以及相机自身那个轴朝上。
//    pangolin::OpenGlRenderState s_cam(
//        pangolin::ProjectionMatrix(752,480,420,420,320,240,0.2,100),
//        pangolin::ModelViewLookAt(-2,2,-2, 0,0,0, pangolin::AxisY)
//    );
//
//    // Create Interactive View in window
//    // CreateDisplay()创建交互式视图，用于显示观察相机“拍摄”到的内容
//    // SetBounds前四个参数依次表示视图在视窗中的范围（下上左右），可以采用相对坐标（0~1）和绝对坐标(使用Attach对象)
//    const int UI_WIDTH = 180;
//    pangolin::Handler3D handler(s_cam);
//
//    // 左侧绘制控制面板
//    pangolin::CreatePanel("ui")
//        .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));
//    // 右侧绘制视窗
//    // pangolin::View& d_cam = pangolin::CreateDisplay()
//    pangolin::View& d_cam = pangolin::Display("cam")
//            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -752.0f/480.0f)
//            .SetHandler(&handler);
//    pangolin::View& cv_img1 = pangolin::Display("img_1")
//            .SetBounds(0.0, 1/3.0f, 1/3.0f, 2/3.0f, 752.0f/480.0f)
//            .SetLock(pangolin::LockRight, pangolin::LockBottom);
//    pangolin::View& cv_img2 = pangolin::Display("img_2")
//            .SetBounds(0.0, 1/3.0f, 2/3.0f, 1.f, 752.0f/480.0f)
//            .SetLock(pangolin::LockRight, pangolin::LockBottom);
//
//    // 控制面板中的控件对象
//    pangolin::Var<bool> A_Button("ui.a_button", false, false); // 按钮
//    pangolin::Var<bool> A_Checkbox("ui.a_checkbox", false, false); // 选框
//    pangolin::Var<double> Double_Slider("ui.a_slider", 3, 0, 5); // double滑条
//    pangolin::Var<int> Int_Slider("ui.b_slider", 2, 0, 5); //int滑条
//    pangolin::Var<std::string> A_string("ui.a_string", "Hello Pangolin");
//    pangolin::Var<bool> SAVE_IMG("ui.save_img", false, false);
//    pangolin::Var<bool> SAVE_WIN("ui.save_win", false, false);
//    pangolin::Var<bool> RECORD_WIN("ui.record_win", false, false);
//    pangolin::Var<std::function<void()> > reset("ui.Reset", SampleMethod);
//
//    // 绑定键盘快捷键
//    pangolin::RegisterKeyPressCallback(pangolin::PANGO_CTRL + 'b', pangolin::SetVarFunctor<double>("ui.a_slider", 3.5));
//    pangolin::RegisterKeyPressCallback(pangolin::PANGO_CTRL + 'r', SampleMethod);
//
//    pangolin::GlTexture imgTexture1(752, 480, GL_RGB, false, 0, GL_BGR, GL_UNSIGNED_BYTE);
//    pangolin::GlTexture imgTexture2(752, 480, GL_RGB, false, 0, GL_BGR, GL_UNSIGNED_BYTE);
//
//    while( !pangolin::ShouldQuit() )
//    {
//        // Clear screen and activate view to render into
//        // 开始绘制时，首先用glClear分别清空色彩缓存和深度缓存
//        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//
//        if(pangolin::Pushed(A_Button)){
//            std::cout << "Push button A." << std::endl;
//        }
//        if(A_Checkbox)
//            Int_Slider = Double_Slider;
//        if(pangolin::Pushed(SAVE_WIN))
//            pangolin::SaveWindowOnRender("window");
//        if(pangolin::Pushed(SAVE_IMG))
//            d_cam.SaveOnRender("cube");
//        // if(pangolin::Pushed(RECORD_WIN))
//        //     pangolin::DisplayBase().RecordOnRender("ffmpeg:[fps=50,bps=8388608,unique_filename]//screencap.avi");
//            // pangolin::RecordOnRender("ffmpeg:[fps=50,bps=8388608,unique_filename]//screencap.avi");
//
//
//
//        // cv::Mat img1 = cv::imread("/home/aal/dataset/slam/EuRoC/MH_01_easy/mav0/cam0/data/1403636579763555584.png");
//        // cv::Mat img2 = cv::imread("/home/aal/dataset/slam/EuRoC/MH_01_easy/mav0/cam0/data/1403636580013555456.png");
//        cv::Mat img1 = cv::imread("/home/aal/dataset/slam/EuRoC/MH_01_easy/mav0/cam0/data/1403636579763555584.png");
//        cv::Mat img2 = cv::imread("/home/aal/dataset/slam/EuRoC/MH_01_easy/mav0/cam0/data/1403636579813555456.png");
//        imgTexture1.Upload(img1.data, GL_BGR, GL_UNSIGNED_BYTE);
//        imgTexture2.Upload(img2.data, GL_BGR, GL_UNSIGNED_BYTE);
//        cv_img1.Activate();
//        glColor3f(1.0f, 1.0f, 1.0f);
//        imgTexture1.RenderToViewportFlipY();
//        cv_img2.Activate();
//        glColor3f(1.0f, 1.0f, 1.0f);
//        imgTexture2.RenderToViewportFlipY();
//
//        // 激活之前设定好的视窗对象
//        d_cam.Activate(s_cam);
//        // Render OpenGL Cube
//        pangolin::glDrawColouredCube();
//
//        // 绘制坐标轴
//        glLineWidth(3);
//        glBegin ( GL_LINES );
//	    glColor3f ( 0.8f,0.f,0.f );
//	    glVertex3f( -1,-1,-1 );
//	    glVertex3f( 0,-1,-1 );
//	    glColor3f( 0.f,0.8f,0.f);
//	    glVertex3f( -1,-1,-1 );
//	    glVertex3f( -1,0,-1 );
//	    glColor3f( 0.2f,0.2f,1.f);
//	    glVertex3f( -1,-1,-1 );
//	    glVertex3f( -1,-1,0 );
//	    glEnd();
//
//        // Swap frames and Process Events
//        pangolin::FinishFrame();
//    }
//
//    // unset the current context from the main thread
//    pangolin::GetBoundWindow()->RemoveCurrent();
//}


int main(int argc, char** argv){
//    setup();

    // use the context in a separate rendering thread
//    std::thread render_loop;
//    render_loop = std::thread(run);
//    render_loop.join();

    cv::Mat img = cv::imread("/home/aal/workspace/dataset/EuRoC/mav0/cam0/data/1403636579763555584.png");
    cv::Mat img1 = cv::imread("/home/aal/workspace/dataset/EuRoC/mav0/cam0/data/1403636580063555584.png");
    std::cout << "imgsize:" <<  img.size() << std::endl;
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
    Naive_SLAM::ORBextractor* orb = new Naive_SLAM::ORBextractor(500, 1.2, 1, 20, 7);
    (*orb)(img, cv::Mat(), keyPoints, descriptions);
    // draw(img, keyPoints, "show_orb");

    std::vector<cv::KeyPoint> keyPoints1;
    cv::Mat descriptions1;
    Naive_SLAM::ORBextractor* orb1 = new Naive_SLAM::ORBextractor(500, 1.2, 1, 20, 7);
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
    std::cout << "l0 pts num: " << pts.size() << std::endl;

    std::chrono::system_clock::time_point t2 = std::chrono::system_clock::now();
    cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(img, img1, pts, pts1, status, err, cv::Size(21, 21), 3, criteria);
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
    DrawMatches(img, img1, ptsMch, pts1Mch);
    
    return 0;
}