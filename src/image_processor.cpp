#include "image_processor.hpp"

namespace image_processor {

void process(const cv::Mat& input, cv::Mat& output, ProcessParams& params) {
    if (input.channels() == 3) {
        cv::cvtColor(input, output, cv::COLOR_BGR2GRAY);
    } else {
        output = input.clone();
    }

    auto clahe = cv::createCLAHE(params.clahe.clipLimit, params.clahe.tilegridSize);
    clahe->apply(output, output);

    cv::GaussianBlur(output, output,
        params.gaussian.ksize,
        params.gaussian.sigmaX,
        params.gaussian.sigmaY,
        params.gaussian.borderType
    );
}

}
