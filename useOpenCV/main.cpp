#include <yaml-cpp/yaml.h>

#include <cassert>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

int main(int argc, char** argv) {
  cv::Mat image1, image2;
  image1 = cv::imread(argv[1], 0);
  image2 = cv::imread(argv[2], 0);
  assert(image1.data != nullptr && image2.data != nullptr);

  // 1. 遍历图像
  cv::Mat image1_clone = image1.clone();
  for (size_t y = 0; y < image1_clone.rows; y++) {
    unsigned char* row_ptr = image1_clone.ptr<unsigned char>(y);
    for (size_t x = 0; x < image1_clone.cols; x++) {
      row_ptr[x] = 255 - row_ptr[x];
    }
  }
  cv::imwrite("1_clone.png", image1_clone);

  // read yaml
  YAML::Node config;
  try {
    config = YAML::LoadFile(argv[3]);
  } catch (YAML::BadFile& e) {
    std::cout << "read error!" << std::endl;
    return -1;
  }
  auto intrinsics = config["intrinsics"].as<std::vector<float>>();
  auto distortion_coefficients = config["distortion_coefficients"].as<std::vector<float>>();
  auto T_BS = config["T_BS"]["data"].as<std::vector<double>>();

  const cv::Mat K = (cv::Mat_<float>(3, 3) << intrinsics[0], 0, intrinsics[2], 0, intrinsics[1],
                     intrinsics[3], 0, 0, 1);
  const cv::Mat D = cv::Mat(distortion_coefficients);
  // std::cout << K << std::endl;
  // std::cout << D << std::endl;

  // 2. 去畸变
  // 2.1 use undistort
  cv::Mat image1_undistorted;
  cv::undistort(image1, image1_undistorted, K, D, K);
  cv::imwrite("1_undistorted.png", image1_undistorted);

  // 2.2 use getOptimalNewCameraMatrix + initUndistortRectifyMap + remap
  cv::Mat map1, map2, image1_remaped, image2_remaped;
  cv::Size imageSize = image1.size();
  const double alpha = 0;
  cv::Mat NewCameraMatrix = cv::getOptimalNewCameraMatrix(K, D, imageSize, alpha, imageSize, 0);
  initUndistortRectifyMap(K, D, cv::Mat(), NewCameraMatrix, imageSize, CV_32FC1, map1, map2);
  remap(image1, image1_remaped, map1, map2, cv::INTER_LINEAR);
  remap(image2, image2_remaped, map1, map2, cv::INTER_LINEAR);
  cv::imwrite("1_remaped.png", image1_remaped);
  cv::imwrite("2_remaped.png", image2_remaped);

  // 3 特征提取、匹配、RANSAC
  // 3.1 提取
  cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(500);
  std::vector<cv::KeyPoint> keypoint1, keypoint2;
  detector->detect(image1_remaped, keypoint1);
  detector->detect(image2_remaped, keypoint2);
  cv::Mat descriptors1, descriptors2;
  cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
  descriptor->compute(image1_remaped, keypoint1, descriptors1);
  descriptor->compute(image2_remaped, keypoint2, descriptors2);

  // 3.2 匹配
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(4);
  std::vector<cv::DMatch> matches;
  matcher->match(descriptors1, descriptors2, matches);
  cv::Mat img_matches;
  cv::drawMatches(image1_remaped, keypoint1, image2_remaped, keypoint2, matches, img_matches);
  cv::imwrite("误匹配消除前.png", img_matches);

  // 3.3 RANSAC
  std::vector<cv::KeyPoint> R_keypoint01, R_keypoint02;
  for (size_t i = 0; i < matches.size(); i++) {
    R_keypoint01.push_back(keypoint1[matches[i].queryIdx]);
    R_keypoint02.push_back(keypoint2[matches[i].trainIdx]);
  }
  std::vector<cv::Point2f> p01, p02;
  for (size_t i = 0; i < matches.size(); i++) {
    p01.push_back(R_keypoint01[i].pt);
    p02.push_back(R_keypoint02[i].pt);
  }

  std::vector<uchar> RansacStatus;
  cv::Mat Fundamental = cv::findFundamentalMat(p01, p02, RansacStatus, cv::FM_RANSAC, 2.);

  std::vector<cv::KeyPoint> RR_keypoint01, RR_keypoint02;
  std::vector<cv::DMatch> RR_matches;
  int index = 0;
  for (size_t i = 0; i < matches.size(); i++) {
    if (RansacStatus[i] != 0) {
      RR_keypoint01.push_back(R_keypoint01[i]);
      RR_keypoint02.push_back(R_keypoint02[i]);
      matches[i].queryIdx = index;
      matches[i].trainIdx = index;
      RR_matches.push_back(matches[i]);
      index++;
    }
  }
  cv::Mat img_RR_matches;
  drawMatches(image1_remaped, RR_keypoint01, image2_remaped, RR_keypoint02, RR_matches,
              img_RR_matches);
  cv::imwrite("消除误匹配点后.png", img_RR_matches);

  return 0;
}