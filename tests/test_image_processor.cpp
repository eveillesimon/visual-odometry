#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include "image_processor.hpp"

namespace {

cv::Mat make_color_gradient_image() {
    cv::Mat image(32, 32, CV_8UC3);

    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            image.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<uchar>((x * 7) % 256),
                static_cast<uchar>((y * 9) % 256),
                static_cast<uchar>(((x + y) * 5) % 256)
            );
        }
    }

    return image;
}

cv::Mat make_grayscale_gradient_image() {
    cv::Mat image(32, 32, CV_8UC1);

    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            image.at<uchar>(y, x) = static_cast<uchar>((x * 8 + y * 3) % 256);
        }
    }

    return image;
}

} // namespace

TEST(ImageProcessor, ProcessConvertsColorInputToSingleChannelOutput) {
    const cv::Mat input = make_color_gradient_image();
    cv::Mat output;
    image_processor::ProcessParams params;

    image_processor::process(input, output, params);

    EXPECT_FALSE(output.empty());
    EXPECT_EQ(output.rows, input.rows);
    EXPECT_EQ(output.cols, input.cols);
    EXPECT_EQ(output.channels(), 1);
    EXPECT_EQ(output.type(), CV_8UC1);
}

TEST(ImageProcessor, ProcessSupportsGrayscaleInput) {
    const cv::Mat input = make_grayscale_gradient_image();
    cv::Mat output;
    image_processor::ProcessParams params;

    image_processor::process(input, output, params);

    EXPECT_FALSE(output.empty());
    EXPECT_EQ(output.rows, input.rows);
    EXPECT_EQ(output.cols, input.cols);
    EXPECT_EQ(output.channels(), 1);
    EXPECT_EQ(output.type(), input.type());
}

TEST(ImageProcessor, ProcessParametersAffectResult) {
    const cv::Mat input = make_color_gradient_image();
    cv::Mat softly_processed;
    cv::Mat strongly_processed;

    image_processor::ProcessParams soft_params;
    soft_params.clahe.clipLimit = 2.0;
    soft_params.clahe.tilegridSize = cv::Size(8, 8);
    soft_params.gaussian.ksize = cv::Size(3, 3);
    soft_params.gaussian.sigmaX = 0.5;
    soft_params.gaussian.sigmaY = 0.5;

    image_processor::ProcessParams strong_params;
    strong_params.clahe.clipLimit = 8.0;
    strong_params.clahe.tilegridSize = cv::Size(4, 4);
    strong_params.gaussian.ksize = cv::Size(9, 9);
    strong_params.gaussian.sigmaX = 3.0;
    strong_params.gaussian.sigmaY = 3.0;

    image_processor::process(input, softly_processed, soft_params);
    image_processor::process(input, strongly_processed, strong_params);

    EXPECT_FALSE(softly_processed.empty());
    EXPECT_FALSE(strongly_processed.empty());
    EXPECT_NE(cv::norm(softly_processed, strongly_processed, cv::NORM_L1), 0.0);
}
