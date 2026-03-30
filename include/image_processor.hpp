#pragma once

#include <opencv2/opencv.hpp>

namespace image_processor {

struct ClaheParams {
    double clipLimit{40.0};
    cv::Size tilegridSize{cv::Size(8, 8)};
};

struct GaussianBlurParams {
    cv::Size ksize{cv::Size(5, 5)};
    double sigmaX{3.0};
    double sigmaY{0.0};
    int borderType{cv::BORDER_DEFAULT};
};

struct ProcessParams {
    ClaheParams clahe{};
    GaussianBlurParams gaussian{};
};


void process(const cv::Mat& input, cv::Mat& output, ProcessParams& params);
    

}