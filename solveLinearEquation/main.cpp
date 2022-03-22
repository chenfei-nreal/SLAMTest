#include <Eigen/Dense>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;

void Loadtxt(const string &strFile, vector<double> &vector_x, vector<double> &vector_y) {
  ifstream f;
  f.open(strFile.c_str());
  assert(f.is_open());

  string s0;
  getline(f, s0);

  while (!f.eof()) {
    string s;
    getline(f, s);
    if (!s.empty()) {
      stringstream ss;
      ss << s;
      double x, y;
      ss >> x;
      vector_x.push_back(x);
      ss >> y;
      vector_y.push_back(y);
    }
  }
}

void calConditionNumber() {}

void getmatrix(const string &strFile, Eigen::Matrix<double, 100, 2> &A,
               Eigen::Matrix<double, 100, 1> &b) {
  vector<double> xs;
  vector<double> ys;
  xs.reserve(100);
  ys.reserve(100);

  // load data.txt
  Loadtxt(strFile, xs, ys);

  // build the least squares equation Ax=b
  A.setOnes();

  for (int i = 0; i < xs.size(); i++) {
    A(i, 0) = xs[i];
    b(i) = ys[i];
  }
}

int main(int argc, char **argv) {
  string strFile = "./data.txt";
  Eigen::Matrix<double, 100, 2> A;
  Eigen::Matrix<double, 100, 1> b;

  getmatrix(strFile, A, b);
  // 1 求解线性方程
  // 1.1 Solve equation with QR decomposition
  Eigen::Vector2d resQr = A.colPivHouseholderQr().solve(b);
  cout << setprecision(16) << endl
       << "The solution with QR decomposition is" << endl
       << resQr << endl;

  // 1.2 Solve equation with SVD decomposition
  Eigen::Vector2d resSVD = A.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(b);
  cout << setprecision(16) << endl
       << "The solution with SVD decomposition is" << endl
       << resSVD << endl;

  // 1.3 Solve equation with
  Eigen::Vector2d resNE = (A.transpose() * A).ldlt().solve(A.transpose() * b);
  cout << setprecision(16) << endl
       << "The solution with normal equation is" << endl
       << resNE << endl;

  // 2 数值稳定性分析
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector2d sv = svd.singularValues();
  std::cout << std::endl;
  std::cout << "data1 系数矩阵条件数：" << std::endl << sv[0] / sv[1] << std::endl << std::endl;

  string strFile2 = "./data2.txt";
  Eigen::Matrix<double, 100, 2> A2;
  Eigen::Matrix<double, 100, 1> b2;
  getmatrix(strFile2, A2, b2);

  Eigen::JacobiSVD<Eigen::MatrixXd> svd2(A2, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector2d sv2 = svd2.singularValues();
  std::cout << "data2 系数矩阵条件数：" << std::endl << sv2[0] / sv2[1] << std::endl << std::endl;

  return 0;
}