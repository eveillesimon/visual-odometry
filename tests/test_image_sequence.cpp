#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include <stdexcept>
#include <string>

#include "image_sequence.hpp"

namespace {

std::string get_assets_dir() {
    return std::string(PROJECT_SOURCE_DIR) + "/tests/assets/";
}

image_io::ImageSequence make_test_sequence() {
    image_io::ImageSequence sequence;
    sequence.directory = get_assets_dir();
    sequence.extension = ".png";
    sequence.first_index = 0;
    sequence.last_index = 3;
    sequence.current_index = 0;
    sequence.zero_padding = 1;
    return sequence;
}

} // namespace

TEST(ImageSequenceTest, BuildImagePathUsesCurrentIndexAndZeroPadding) {
    image_io::ImageSequence sequence;
    sequence.directory = "/tmp/";
    sequence.extension = ".png";
    sequence.current_index = 42;
    sequence.zero_padding = 10;

    const std::string path = image_io::build_image_path(sequence);

    EXPECT_EQ(path, "/tmp/0000000042.png");
}

TEST(ImageSequenceTest, HasNextIsTrueAtBeginningAndFalseAfterEnd) {
    image_io::ImageSequence sequence = make_test_sequence();

    EXPECT_TRUE(image_io::has_next(sequence));

    sequence.current_index = 3;
    EXPECT_TRUE(image_io::has_next(sequence));

    sequence.current_index = 4;
    EXPECT_FALSE(image_io::has_next(sequence));
}

TEST(ImageSequenceTest, ReadNextLoadsRealImagesAndAdvancesCursor) {
    image_io::ImageSequence sequence = make_test_sequence();
    cv::Mat image;

    ASSERT_TRUE(image_io::read_next(sequence, image, cv::IMREAD_COLOR));
    EXPECT_FALSE(image.empty());
    EXPECT_EQ(sequence.current_index, 1);

    ASSERT_TRUE(image_io::read_next(sequence, image, cv::IMREAD_COLOR));
    EXPECT_FALSE(image.empty());
    EXPECT_EQ(sequence.current_index, 2);

    ASSERT_TRUE(image_io::read_next(sequence, image, cv::IMREAD_COLOR));
    EXPECT_FALSE(image.empty());
    EXPECT_EQ(sequence.current_index, 3);
}

TEST(ImageSequenceTest, ReadNextReturnsFalseWhenSequenceIsFinished) {
    image_io::ImageSequence sequence = make_test_sequence();
    cv::Mat image;

    ASSERT_TRUE(image_io::read_next(sequence, image, cv::IMREAD_COLOR));
    ASSERT_TRUE(image_io::read_next(sequence, image, cv::IMREAD_COLOR));
    ASSERT_TRUE(image_io::read_next(sequence, image, cv::IMREAD_COLOR));
    ASSERT_TRUE(image_io::read_next(sequence, image, cv::IMREAD_COLOR));

    EXPECT_FALSE(image_io::read_next(sequence, image, cv::IMREAD_COLOR));
    EXPECT_EQ(sequence.current_index, 4);
}

TEST(ImageSequenceTest, ReadNextCanLoadGrayscale) {
    image_io::ImageSequence sequence = make_test_sequence();
    cv::Mat image;

    ASSERT_TRUE(image_io::read_next(sequence, image, cv::IMREAD_GRAYSCALE));
    EXPECT_FALSE(image.empty());
    EXPECT_EQ(image.channels(), 1);
}