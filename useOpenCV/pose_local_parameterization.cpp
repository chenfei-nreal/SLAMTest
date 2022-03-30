#include "pose_local_parameterization.h"
#include <iostream>

/**
 * @brief
 *
 * @param x [0-2]: t [3-5]: quaternion
 * @param delta [0-2]: delta_t [3-5]: delta_phi
 * @param x_plus_delta
 * @return true
 * @return false
 */

bool QuaternionLocalParameterization::Plus(const double *x, const double *delta,
                                           double *x_plus_delta) const {
  Eigen::Map<const Eigen::Quaterniond> _q(x);
  Eigen::Quaterniond dq = deltaQ(Eigen::Map<const Eigen::Vector3d>(delta));
  Eigen::Map<Eigen::Quaterniond> q(x_plus_delta);

  q = (_q * dq).normalized();

  return true;
}

bool QuaternionLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const {
  Eigen::Map<Eigen::Matrix<double, 4, 6, Eigen::RowMajor>> j(jacobian);
  j.topRows<3>().setIdentity();
  j.bottomRows<1>().setZero();

  return true;
}