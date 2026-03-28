# Visual Odometry 

## Overview 

This project implements a monocular **visual odometry pipeline** in C++ and is based on OpenCV.

Its goal is to estimate the movements of a camera based on a sequence of images by :
- detecting features 
- tracking them between frames
- filtering unreliable matches

This project is a first step toward a complete SLAM / Visual Odometry system.

## Pipeline

1. Image processing : 

To enhance feature detectors accuracy, I perform adaptive histogram equalization (`cv::CLAHE`) and gaussian blurring to reduce signal noise.

 
2. Feature detection :

To detect new points of interest in the image, I use the Shi-Thomasi method (`cv::goodFeaturesToTrack()`).


3. Tracking features : 

The following of the previously discovered features is made using a sparse optical flow approach. The Lucas-Kanade method (`cv::calcOpticalFlowPyrLK()`) is based on retrieving small displacement of patches of pixels using a corner detector and a similarity check. 


1. Filtering outliers :

The optical flow methods can produce some unreliable tracking. These outliers are removed based on a measure of the distance between where they were supposed to be, and where they really are.


5. Adding new features :

Until now, the total number of studied features is decreasing (because of points that could not be tracked and outliers removed). When the total number of feature goes below a threshold, a new Shi-Thomasi iteration is done to ensure sufficient feature tracking.    



## Installation

### Ubuntu :

1. Requirements

Install required build and compile tools, as well as OpenCV: 

```
sudo apt update
sudo apt upgrade
sudo apt install -y build-essential cmake pkg-config libopencv-dev
```


2. Add data

**TODO :** Add proper way to open data (sequence of dated pictures or video)

Currently using KITTI-360 perspecctive camera images.  



3. Building

Once the project is cloned, add the `/visual-odometry/build` directory.

Then from the project's root. Resolve dependencies and generate the Makefile: 
```
cmake -S . -B build -DBUILD_TESTING=ON
```

Obviously, you can choose to disable the tests.



Compile the source files:
```
cmake --build build -j"$(nproc)"
```

4. Execute the project

```
./build/visual_odometry
```

Or the tests:
```
ctest --test-dir build --output-on-failure
```
