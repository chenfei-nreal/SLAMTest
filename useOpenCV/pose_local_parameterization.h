#pragma once

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "rotation.h"


class SO3LocalParameterization : public ceres::LocalParameterization {
  virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
  virtual bool ComputeJacobian(const double *x, double *jacobian) const;
  virtual int GlobalSize() const { return 9; };
  virtual int LocalSize() const { return 3; };
};

class QuaternionLocalParameterization : public ceres::LocalParameterization {
  virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
  virtual bool ComputeJacobian(const double *x, double *jacobian) const;
  virtual int GlobalSize() const { return 4; };
  virtual int LocalSize() const { return 3; };
};
