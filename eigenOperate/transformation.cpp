#include <Eigen/Geometry>
#include <iostream>

Eigen::Matrix3d skew(const Eigen::Vector3d &w) {
  Eigen::Matrix3d skew_w;
  skew_w << 0, -w(2), w(1), w(2), 0, -w(0), -w(1), w(0), 0;
  return skew_w;
}

/**
 * @brief Mutiply two JPL quaternions
 *
 * @param q JPL quaternion
 * @param p JPL quaternion
 * @return Eigen::Matrix<double, 4, 1> q*p quaternion
 */

Eigen::Matrix<double, 4, 1> quat_multiply(const Eigen::Matrix<double, 4, 1> &q,
                                          const Eigen::Matrix<double, 4, 1> &p) {
  Eigen::Matrix<double, 4, 1> q_t;
  Eigen::Matrix<double, 4, 4> q_L;

  // Quaternion kinematics for the error-state Kalman filter
  // Indirect Kalman Filter for 3D Attitude Estimation
  // q⊗p = [q]L * p
  q_L.block(0, 0, 3, 3) = q(3, 0) * Eigen::MatrixXd::Identity(3, 3) - skew(q.block(0, 0, 3, 1));
  q_L.block(0, 3, 3, 1) = q.block(0, 0, 3, 1);
  q_L.block(3, 0, 1, 3) = -q.block(0, 0, 3, 1).transpose();
  q_L(3, 3) = q(3, 0);

  q_t = q_L * p;

  // ensure unique by forcing q_4 to be > 0
  if (q_t(3, 0) < 0) {
    q_t(3, 0) *= -1;
  }

  return q_t / q_t.norm();
}

int main(int argc, char **argv) {
  Eigen::Vector4d left_imu_q_cam;
  Eigen::Vector3d left_imu_p_cam;
  left_imu_q_cam << 0.13431639597354814, 0.00095051670014565813, -0.0084222184858180373,
    0.99090224973327068;
  left_imu_p_cam << -0.050720060477640147, -0.0017414170413474165, 0.0022943667597148118;

  Eigen::Vector4d right_imu_q_cam;
  Eigen::Vector3d right_imu_p_cam;
  right_imu_q_cam << 0.13492462817073628, -0.00013648999867379373, -0.015306242884176362,
    0.99073762672679389;
  right_imu_p_cam << 0.051932496584961352, -0.0011555929083120534, 0.0030949732069645722;

  Eigen::Vector4d left_imu_q_cam_conjugate = left_imu_q_cam;
  left_imu_q_cam_conjugate.head(3) *= -1;

  // 双目外参
  Eigen::Vector4d left_q_right = quat_multiply(left_imu_q_cam_conjugate, right_imu_q_cam);
  Eigen::Vector3d left_p_right = right_imu_p_cam - left_imu_p_cam;

  std::cout << "left_q_right is :" << std::endl << left_q_right << std::endl << std::endl;
  std::cout << "left_p_right is :" << std::endl << left_p_right << std::endl << std::endl;

  return 0;
}