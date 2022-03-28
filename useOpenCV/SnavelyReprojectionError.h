#pragma once

#include <iostream>
#include "ceres/ceres.h"
#include "rotation.h"

class SnavelyReprojectionError {
 public:
  SnavelyReprojectionError(double observation_x, double observation_y)
      : observed_x(observation_x), observed_y(observation_y) {}

  template <typename T>
  bool operator()(const T* const camera, const T* const point, T* residuals) const {
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
   * @param camera [0-2] angle-axis [3-5] translation [6-9] fx fy u0 v0
   * @param point 3D location
   * @param predictions 2D predictions
   * @return true
   * @return false
   */
  template <typename T>
  static inline bool CamProjection(const T* camera, const T* point, T* predictions) {
    T p[3];
    AngleAxisRotatePoint(camera, point, p);

    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];

    T xp = p[0] / p[2];
    T yp = p[1] / p[2];

    const T& fx = camera[6];
    const T& fy = camera[7];
    const T& u0 = camera[8];
    const T& v0 = camera[9];

    predictions[0] = fx * xp + u0;
    predictions[1] = fy * yp + v0;

    return true;
  }

  static ceres::CostFunction* Create(const double observed_x, const double observed_y) {
    return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 10, 3>(
      new SnavelyReprojectionError(observed_x, observed_y)));
  }

 private:
  double observed_x;
  double observed_y;
};
