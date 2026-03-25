#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>



const std::string PROJECT_DIR = "/home/simon/Workspace/visual-odometry/";
const std::string IMAGE_DIR = "data/KITTI-360/data_2d_raw/2013_05_28_drive_0004_sync/image_00/data_rect/";
constexpr int FIRST_IMAGE = 1000;
constexpr int LAST_IMAGE  = 10000;


std::string get_image_path(const int image_id, const std::string& abs_image_dir) {
    std::string name = std::to_string(image_id);
    std::string zeroes(10 - name.length(), '0');
    std::string extension = ".png";
    return abs_image_dir + zeroes + name + extension;
}

void display_corners(cv::Mat& img, const std::vector<cv::Point2f>& corners, const std::vector<uchar>& status) {

    for (size_t i = 0; i < corners.size(); i++) {
        if (status[i]== 1) {
            cv::circle(img, corners[i], 3, cv::Scalar(50, 50, 230), cv::FILLED);
        }
    }
}

void display_corners(cv::Mat& img, const std::vector<cv::Point2f>& corners) {
    auto status = std::vector<uchar>(corners.size(), 1);
    display_corners(img, corners, status);
}

void display_features_movement(cv::Mat& img, const std::vector<cv::Point2f>& previous_corners, const std::vector<cv::Point2f>& current_corners) {
    for (size_t i = 0; i < current_corners.size(); i++) {
        cv::line(img, previous_corners[i], current_corners[i], cv::Scalar(50, 50, 250));
    }
}

void prepare_for_feature_detection(const cv::Mat& input, cv::Mat& output) {
    cv::cvtColor(input, output, cv::COLOR_BGR2GRAY);    // convert image to grayscale

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(output, output);                       // histogram equalization using adaptative CLAHE approach

    cv::GaussianBlur(output, output, cv::Size(3,3), 0); // gaussian blur to reduce noise
}

void filter_outlier_features(
    const cv::Mat& img, std::vector<cv::Point2f>& raw_features, 
    const std::vector<cv::Point2f>& previous_features, 
    std::vector<cv::Point2f>& filtered_features, 
    std::vector<cv::Point2f>& previous_filtered_features, 
    const std::vector<uchar>& status, 
    const std::vector<float>& errors
) {
    for (size_t i = 0; i < raw_features.size(); i++) {
        
        const cv::Point2f p = raw_features[i];
        const cv::Point2f prev_p = previous_features[i];

        if (
            p.x < 0 || p.x >= img.cols ||
            p.y < 0 || p.y >= img.rows
        ) {
            continue;
        }

        if (status[i] && errors[i] < 20.0f) {
            filtered_features.push_back(p);
            previous_filtered_features.push_back(prev_p);            
        }
    }
}


int main() {

    const std::string abs_image_dir = PROJECT_DIR + IMAGE_DIR;


    // Loop initialization
    cv::Mat previous_image = cv::imread(get_image_path(FIRST_IMAGE, abs_image_dir), cv::IMREAD_COLOR);
    cv::Mat previous_image_processed;
    prepare_for_feature_detection(previous_image, previous_image_processed);
 
    std::vector<cv::Point2f> previous_features;
    cv::goodFeaturesToTrack(previous_image_processed, previous_features, 500, 0.01, 5);   // last params: number of features, quality level, min. euclidian distance

    for (int i = FIRST_IMAGE + 1; i <= LAST_IMAGE; i++) {
        cv::Mat current_image = cv::imread(get_image_path(i, abs_image_dir), cv::IMREAD_COLOR);

        if (current_image.empty()) {
            std::cerr << "Failed to load image " << get_image_path(i, abs_image_dir) << std::endl;
            break;
        }

        cv::Mat current_image_processed;
        prepare_for_feature_detection(current_image, current_image_processed);

        cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 400, 0.01);
        std::vector<uchar> status;
        std::vector<float> errors;
        std::vector<cv::Point2f> raw_features;
        cv::calcOpticalFlowPyrLK(previous_image_processed, current_image_processed, previous_features, raw_features, status, errors, cv::Size(31, 31), 5, criteria);
        
        std::vector<cv::Point2f> filtered_features;
        std::vector<cv::Point2f> previous_filtered_features; 
        filter_outlier_features(current_image_processed, raw_features, previous_features, filtered_features, previous_filtered_features, status, errors);
        
        
        // Managing interface with mask to avoid changing original image
        cv::Mat current_overlay = cv::Mat::zeros(current_image.size(), current_image.type());
        cv::Mat mask, current_overlay_gray;
        display_features_movement(current_overlay, previous_filtered_features, filtered_features);
        cv::cvtColor(current_overlay, current_overlay_gray, cv::COLOR_BGR2GRAY);
        cv::threshold(current_overlay_gray, mask, 1, 255, cv::THRESH_BINARY_INV);
        current_image.copyTo(current_overlay, mask);

        previous_features = filtered_features;

        // add new corners when too few
        std::vector<cv::Point2f> new_features;
        if (previous_features.size() < 400) {
            cv::goodFeaturesToTrack(current_image_processed, new_features, 1000, 0.00001, 5);
            
            for (const auto& feat: new_features) {
                previous_features.push_back(feat);
            }
        }
        

        // Displaying 
        cv::Mat display, current_image_processed_bgr;
        cv::cvtColor(current_image_processed, current_image_processed_bgr, cv::COLOR_GRAY2BGR);
        
        cv::Mat processed_overlay = cv::Mat::zeros(current_image_processed_bgr.size(), current_image_processed_bgr.type());
        cv::Mat processed_mask, processed_overlay_gray;
        display_corners(processed_overlay, new_features);
        cv::cvtColor(processed_overlay, processed_overlay_gray, cv::COLOR_BGR2GRAY);
        cv::threshold(processed_overlay_gray, processed_mask, 1, 255, cv::THRESH_BINARY_INV);
        current_image_processed_bgr.copyTo(processed_overlay, processed_mask);

    
        cv::vconcat(current_overlay, processed_overlay, display);
        cv::imshow("Display features prev/current", display);

        previous_image_processed = current_image_processed.clone();

        cv::waitKey(0);

    }

    
    return 0;
}