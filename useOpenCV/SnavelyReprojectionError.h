#pragma once

#include <Eigen/Dense>
#include <iostream>
#include "ceres/ceres.h"
#include "rotation.h"

class SnavelyReprojectionError {
 public:
  SnavelyReprojectionError(double observation_x, double observation_y, const double fx,
                           const double fy, const double u0, const double v0)
      : observed_x(observation_x), observed_y(observation_y), fx(fx), fy(fy), u0(u0), v0(v0) {}

  template <typename T>
  bool operator()(const T *const camera, const T *const point, T *residuals) const {
    T predictions[2];
    CamProjection(camera, point, predictions);
    residuals[0] = predictions[0] - T(observed_x);
    residuals[1] = predictions[1] - T(observed_y);

    return true;
  }

  /**
   * @brief
   *
   * @tparam T
   * @param camera [0-2] angle-axis [3-5] translation
   * @param point 3D location
   * @param predictions 2D predictions
   * @return true
   * @return false
   */
  template <typename T>
  inline bool CamProjection(const T *camera, const T *point, T *predictions) const {
    T p[3];
    AngleAxisRotatePoint(camera, point, p);

    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];

    T xp = p[0] / p[2];
    T yp = p[1] / p[2];

    predictions[0] = fx * xp + u0;
    predictions[1] = fy * yp + v0;

    return true;
  }

  static ceres::CostFunction *Create(const double observed_x, const double observed_y,
                                     const double fx, const double fy, const double u0,
                                     const double v0) {
    return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 6, 3>(
      new SnavelyReprojectionError(observed_x, observed_y, fx, fy, u0, v0)));
  }

 private:
  double fx, fy, u0, v0;
  double observed_x;
  double observed_y;
};

class customizedCostFunction : public ceres::SizedCostFunction<2, 3, 3, 3> {
 public:
  customizedCostFunction(const double x, const double y, const double fx, const double fy,
                         const double u0, const double v0)
      : observed_x(x), observed_y(y), fx(fx), fy(fy), u0(u0), v0(v0) {}
  virtual ~customizedCostFunction() {}

  /**
   * @brief this evaluates the error term and additionally computes the
   * Jacobians.
   *
   * @param parameters Pointer to the parameters
   * @param residuals Pointer to the residual vector
   * @param jacobians Pointer to the Jacobians
   * @return true
   * @return false
   */
  virtual bool Evaluate(double const *const *parameters, double *residuals,
                        double **jacobians) const {
    // camera pose
    Eigen::Map<const Eigen::Vector3d> rotation_vector(&parameters[0][0]);
    const Eigen::Matrix3d R = RotationVectorToRotationMatrix(rotation_vector);
    Eigen::Map<const Eigen::Vector3d> t(&parameters[1][0]);

    // point in world coordinates
    Eigen::Map<const Eigen::Vector3d> P(&parameters[2][0]);

    // point in camera coordinates
    Eigen::Vector3d Pc = R * P + t;

    double predictions[2];

    double xp = Pc[0] / Pc[2];
    double yp = Pc[1] / Pc[2];

    predictions[0] = fx * xp + u0;
    predictions[1] = fy * yp + v0;

    // observation - prediction
    residuals[0] = observed_x - predictions[0];
    residuals[1] = observed_y - predictions[1];

    if (!jacobians) {
      return true;
    }

    Eigen::Matrix<double, 2, 3> partial_e_partial_Pc;
    partial_e_partial_Pc << -fx / Pc[2], 0, (fx * Pc[0]) / (Pc[2] * Pc[2]), 0, -fy / Pc[2],
      (fy * Pc[1]) / (Pc[2] * Pc[2]);

    if (jacobians[0] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J0(jacobians[0]);
      J0 = -partial_e_partial_Pc * skew(R * P) * RightJacobianSO3(-rotation_vector);
      // J0 = -partial_e_partial_Pc * skew(R * P);
    }

    if (jacobians[1] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J1(jacobians[1]);
      J1 = partial_e_partial_Pc;
    }

    if (jacobians[2] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J2(jacobians[2]);
      J2 = partial_e_partial_Pc * R;
    }

    return true;
  }

 private:
  double fx, fy, u0, v0;
  double observed_x;
  double observed_y;
};