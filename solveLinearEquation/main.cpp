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

int main(int argc, char **argv) {
  string strFile = "./data.txt";
  vector<double> xs;
  vector<double> ys;
  ys.reserve(100);
  ys.reserve(100);

  // load data.txt
  Loadtxt(strFile, xs, ys);

  // build the least squares equation Ax=b
  Eigen::Matrix<double, 100, 2> A;
  Eigen::Matrix<double, 100, 1> b;
  A.setOnes();

  for (int i = 0; i < xs.size(); i++) {
    A(i, 0) = xs[i];
    b(i) = ys[i];
  }

  // 1. Solve equation with QR decomposition
  Eigen::Vector2d resQr = A.colPivHouseholderQr().solve(b);
  cout << setprecision(16) << endl
       << "The solution with QR decomposition is" << endl
       << resQr << endl;

  // 2. Solve equation with SVD decomposition
  Eigen::Vector2d resSVD = A.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(b);
  cout << setprecision(16) << endl
       << "The solution with SVD decomposition is" << endl
       << resSVD << endl;

  // 3. Solve equation with
  Eigen::Vector2d resNE = (A.transpose() * A).ldlt().solve(A.transpose() * b);
  cout << setprecision(16) << endl
       << "The solution with normal equation is" << endl
       << resNE << endl;
}