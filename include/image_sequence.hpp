#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace image_io {

struct ImageSequence {
    std::string directory;
    std::string extension = ".png";
    int first_index = 0;
    int last_index = 0;
    int current_index = 0;
    int zero_padding = 10;
};

std::string build_image_path(const ImageSequence& sequence);
bool has_next(const ImageSequence& sequence);
bool read_next(ImageSequence& sequence,  cv::Mat& image_output, cv::ImreadModes mode);

}