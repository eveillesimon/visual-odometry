#include "image_sequence.hpp"

namespace image_io {

std::string build_image_path(const ImageSequence& sequence) {
    std::string name = std::to_string(sequence.current_index);
    std::string zeroes(10 - name.length(), '0');
    return sequence.directory + zeroes + name + sequence.extension;
}

bool has_next(const ImageSequence& sequence) {
    return sequence.current_index <= sequence.last_index;
}

bool read_next(ImageSequence& sequence, cv::Mat& image_output, cv::ImreadModes mode) {
    if (!has_next(sequence)) {
        return false;
    }

    const std::string path = build_image_path(sequence);
    image_output = cv::imread(path, mode);

    if (image_output.empty()) {
        throw std::runtime_error("Failed to read image: " + path);
    }

    sequence.current_index++;
    return true;
}

}
