#include <yaml-cpp/yaml.h>

#include <cassert>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

int main(int argc, char** argv) {
  cv::Mat image1, image2;
  image1 = cv::imread(argv[1], 0);
  image2 = cv::imread(argv[2], 0);
  assert(image1.data != nullptr && image2.data != nullptr);
  // std::cout << image1.rows << "  " << image1.cols << "  " << image1.channels() << std::endl;

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
  cv::Mat map1, map2, image1_remaped;
  cv::Size imageSize(image1.cols, image1.rows);
  const double alpha = 0;
  cv::Mat NewCameraMatrix = cv::getOptimalNewCameraMatrix(K, D, imageSize, alpha, imageSize, 0);
  initUndistortRectifyMap(K, D, cv::Mat(), NewCameraMatrix, imageSize, CV_16SC2, map1, map2);
  remap(image1, image1_remaped, map1, map2, cv::INTER_LINEAR);
  cv::imwrite("1_remaped.png", image1_remaped);
  // cv::imshow("1.png", image_remaped);
  // cv::waitKey(0);
  return 0;
}