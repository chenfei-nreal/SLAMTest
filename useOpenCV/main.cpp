#include <yaml-cpp/yaml.h>

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <cassert>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include "SnavelyReprojectionError.h"

Eigen::Vector3f triangulate(const cv::KeyPoint &p1, const cv::KeyPoint &p2,
                            const Eigen::Matrix<float, 3, 4> &P1,
                            const Eigen::Matrix<float, 3, 4> &P2);

int checkRT(const Eigen::Matrix3f &R, const Eigen::Vector3f &t, const Eigen::Matrix3f &K,
            const std::vector<cv::DMatch> &matches, const std::vector<cv::KeyPoint> &keypoints1,
            const std::vector<cv::KeyPoint> &keypoints2);

int main(int argc, char **argv) {
  cv::Mat image1, image2;
  image1 = cv::imread(argv[1], 0);
  image2 = cv::imread(argv[2], 0);
  assert(image1.data != nullptr && image2.data != nullptr);

  // 1. 遍历图像
  cv::Mat image1_clone = image1.clone();
  for (size_t y = 0; y < image1_clone.rows; y++) {
    unsigned char *row_ptr = image1_clone.ptr<unsigned char>(y);
    for (size_t x = 0; x < image1_clone.cols; x++) {
      row_ptr[x] = 255 - row_ptr[x];
    }
  }
  cv::imwrite("1_clone.png", image1_clone);

  // read yaml
  YAML::Node config;
  try {
    config = YAML::LoadFile(argv[3]);
  } catch (YAML::BadFile &e) {
    std::cout << "read error!" << std::endl;
    return -1;
  }
  auto intrinsics = config["intrinsics"].as<std::vector<float>>();
  auto distortion_coefficients = config["distortion_coefficients"].as<std::vector<float>>();
  auto T_BS = config["T_BS"]["data"].as<std::vector<double>>();

  const cv::Mat K = (cv::Mat_<float>(3, 3) << intrinsics[0], 0, intrinsics[2], 0, intrinsics[1],
                     intrinsics[3], 0, 0, 1);
  Eigen::Matrix3f K_eigen;
  K_eigen << intrinsics[0], 0, intrinsics[2], 0, intrinsics[1], intrinsics[3], 0, 0, 1;
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

  // 4 八点法求相机pose

  // 4.1 构建方程组，使用SVD分解求基础矩阵
  // 实践中一般需要归一化特征点到统一尺度并做RANSAC迭代，本代码目前只是最小实现
  Eigen::Matrix<float, 8, 9> A;
  Eigen::Matrix<float, 9, 1> x;
  for (int i = 0; i < 8; i++) {
    const float u1 = RR_keypoint01[i].pt.x;
    const float v1 = RR_keypoint01[i].pt.y;
    const float u2 = RR_keypoint02[i].pt.x;
    const float v2 = RR_keypoint02[i].pt.y;

    A(i, 0) = u2 * u1;
    A(i, 1) = u2 * v1;
    A(i, 2) = u2;
    A(i, 3) = v2 * u1;
    A(i, 4) = v2 * v1;
    A(i, 5) = v2;
    A(i, 6) = u1;
    A(i, 7) = v1;
    A(i, 8) = 1;
  }

  Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  auto V = svd.matrixV();
  // 最小奇异值对应的右向量
  Eigen::Matrix<float, 9, 1> Fv = V.col(8);
  Eigen::Matrix<float, 3, 3> Fpre;
  Fpre << Fv(0), Fv(1), Fv(2), Fv(3), Fv(4), Fv(5), Fv(6), Fv(7), Fv(8);
  // std::cout << Fpre << std::endl << std::endl;

  // 基础矩阵秩为2，再次对F进行奇异值分解强制其秩为2
  Eigen::JacobiSVD<Eigen::Matrix3f> svd2(Fpre, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector3f singularvalues2 = svd2.singularValues();
  Eigen::Matrix3f U2 = svd2.matrixU();
  Eigen::Matrix3f V2 = svd2.matrixV();
  Eigen::Matrix3f sigma;
  sigma.setZero();
  sigma(0, 0) = singularvalues2(0);
  sigma(1, 1) = singularvalues2(1);
  Eigen::Matrix3f F = U2 * sigma * V2.transpose();

  // 4.2 本质矩阵分解，恢复R,t
  // Multiple View Geometry in Computer Vision
  Eigen::Matrix3f E = K_eigen.transpose() * F * K_eigen; // E = K.t()*F*K
  Eigen::JacobiSVD<Eigen::Matrix3f> svdE(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3f Ue = svdE.matrixU();
  Eigen::Matrix3f Ve = svdE.matrixV();
  auto singularvaluesE = svdE.singularValues();

  // 左奇异值矩阵U的最后一列是t，对其进行归一化
  Eigen::Vector3f t = Ue.col(2);
  t = t / t.norm();

  // 构建W矩阵
  Eigen::Matrix3f W = Eigen::Matrix3f::Zero();
  W(0, 1) = -1;
  W(1, 0) = 1;
  W(2, 2) = 1;

  Eigen::Matrix3f R1 = Ue * W * Ve.transpose();
  if (R1.determinant() < 0) {
    std::cout << "R1 det is < 0 take the opposite." << std::endl;
    R1 = -R1;
  }

  Eigen::Matrix3f R2 = Ue * W.transpose() * Ve.transpose();
  if (R2.determinant() < 0) {
    std::cout << "R2 det is < 0 take the opposite." << std::endl;
    R2 = -R2;
  }

  std::vector<Eigen::Matrix3f> Rs{R1, R1, R2, R2};
  std::vector<Eigen::Vector3f> ts{t, -t, t, -t};

  int max_value = 0;
  int good_index = -1;
  for (int i = 0; i < 4; i++) {
    // std::cout << Rs[i] << std::endl;
    int nGood = checkRT(Rs[i], ts[i], K_eigen, RR_matches, RR_keypoint01, RR_keypoint02);
    if (nGood > max_value) {
      max_value = nGood;
      good_index = i;
    }
  }

  Eigen::Matrix3f R21 = Rs[good_index];
  Eigen::Vector3f t21 = ts[good_index];

  std::cout << "\n camera pose R21 is \n" << R21 << std::endl;
  std::cout << "\n camera pose t21 is \n" << t21 << std::endl;

  // 5 利用最优的R,t，对2D匹配点做三角化
  // 以第一个相机光心为世界坐标系，即：P1 = K*[I|0]
  Eigen::Matrix<float, 3, 4> P1;
  P1.block<3, 3>(0, 0) = K_eigen;
  P1.col(3) = Eigen::Vector3f::Zero();

  // 第二个相机的投影矩阵 P2 = K*[R|t]
  // 对极约束求解的是R21，t21
  Eigen::Matrix<float, 3, 4> P2;
  P2.block(0, 0, 3, 3) = R21;
  P2.col(3) = t21;
  P2 = K_eigen * P2;

  std::vector<Eigen::Vector3f> p3ds;
  for (int i = 0; i < RR_matches.size(); i++) {
    Eigen::Vector3f p3d = triangulate(RR_keypoint01[RR_matches[i].queryIdx],
                                      RR_keypoint02[RR_matches[i].trainIdx], P1, P2);
    p3ds.push_back(p3d);
  }

  // 6 Use Ceres solver
  ceres::Problem problem;
  const double &fx = K_eigen(0, 0);
  const double &fy = K_eigen(1, 1);
  const double &u0 = K_eigen(0, 2);
  const double &v0 = K_eigen(1, 2);

  Eigen::AngleAxisf angleAxis1(Eigen::Matrix3f::Identity());
  Eigen::AngleAxisf angleAxis2(R21);

  Eigen::Vector3f aa1 = angleAxis1.angle() * angleAxis1.axis();
  Eigen::Vector3f aa2 = angleAxis2.angle() * angleAxis2.axis();

  double camera1[] = {aa1[0], aa1[1], aa1[2], 0, 0, 0, fx, fy, u0, v0};
  double camera2[] = {aa2[0], aa2[1], aa2[2], t21[0], t21[1], t21[2], fx, fy, u0, v0};

  const int points_num = p3ds.size();
  std::cout << std::endl;

  double *point_parameters = new double[points_num * 3];

  for (int i = 0; i < points_num; i++) {
    point_parameters[i * 3 + 0] = p3ds[i][0];
    point_parameters[i * 3 + 1] = p3ds[i][1];
    point_parameters[i * 3 + 2] = p3ds[i][2];
  }

  ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

  for (int i = 0; i < points_num; i++) {
    // ceres::CostFunction *cost_function1 = SnavelyReprojectionError::Create(
    //   double(RR_keypoint01[i].pt.x), double(RR_keypoint01[i].pt.y));
    // problem.AddResidualBlock(cost_function1, loss_function, camera1, point_parameters + i * 3);

    ceres::CostFunction *cost_function2 = SnavelyReprojectionError::Create(
      double(RR_keypoint02[i].pt.x), double(RR_keypoint02[i].pt.y));
    problem.AddResidualBlock(cost_function2, loss_function, camera2, point_parameters + i * 3);
  }

  std::cout << "Solve ceres BA ... " << std::endl;
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::LinearSolverType::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << std::endl;

  delete[] point_parameters;

  return 0;
}

Eigen::Vector3f triangulate(const cv::KeyPoint &p1, const cv::KeyPoint &p2,
                            const Eigen::Matrix<float, 3, 4> &P1,
                            const Eigen::Matrix<float, 3, 4> &P2) {
  Eigen::Matrix4f A;
  A.row(0) = p1.pt.x * P1.row(2) - P1.row(0);
  A.row(1) = p1.pt.y * P1.row(2) - P1.row(1);
  A.row(2) = p2.pt.x * P2.row(2) - P2.row(0);
  A.row(3) = p2.pt.y * P2.row(2) - P2.row(1);

  // SVD
  Eigen::JacobiSVD<Eigen::Matrix4f> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector4f X = svd.matrixV().col(3);
  // 齐次坐标
  return X.head(3) / X[3];
}

int checkRT(const Eigen::Matrix3f &R, const Eigen::Vector3f &t, const Eigen::Matrix3f &K,
            const std::vector<cv::DMatch> &matches, const std::vector<cv::KeyPoint> &keypoints1,
            const std::vector<cv::KeyPoint> &keypoints2) {
  // 以第一个相机光心为世界坐标系，即：P1 = K*[I|0]
  Eigen::Matrix<float, 3, 4> P1;
  P1.block<3, 3>(0, 0) = K;
  P1.col(3) = Eigen::Vector3f::Zero();
  Eigen::Vector3f O1 = Eigen::Vector3f::Identity();
  // std::cout << "P1: " << std::endl << P1 << std::endl;

  // 第二个相机的投影矩阵 P2 = K*[R|t]
  // 对极约束求解的是R21，t21
  Eigen::Matrix<float, 3, 4> P2;
  P2.block(0, 0, 3, 3) = R;
  P2.col(3) = t;
  P2 = K * P2;
  // 第二个相机在第一个相机坐标系下的坐标
  Eigen::Vector3f O2 = -R.transpose() * t;

  int nGood = 0;
  for (size_t i = 0; i < matches.size(); i++) {
    const cv::KeyPoint &kp1 = keypoints1[matches[i].queryIdx];
    const cv::KeyPoint &kp2 = keypoints2[matches[i].trainIdx];

    Eigen::Vector3f X = triangulate(kp1, kp2, P1, P2);

    Eigen::Vector3f normal1 = X - O1;
    float dist1 = normal1.norm();

    Eigen::Vector3f normal2 = X - O2;
    float dist2 = normal2.norm();

    // ab = |a||b|cos_theta
    float cos_theta = (normal1.dot(normal2)) / (dist1 * dist2);
    Eigen::Vector3f XinC2 = R * X + t;

    if (X[2] < 0 || XinC2[2] < 0 || cos_theta > 0.99998) {
      continue;
    }

    nGood++;
  }

  return nGood;
}
