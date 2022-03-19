#include <Eigen/Dense>
#include <iostream>

int main(int argc, char **argv) {
  Eigen::Matrix<float, 4, 4> matrix4f;
  matrix4f << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16;

  /** 1. 块操作 **/
  // 1.1 block用于左值操作
  // 固定大小子矩阵
  std::cout << "Block in the middle" << std::endl;
  std::cout << matrix4f.block<2, 2>(1, 1) << std::endl;
  // 动态大小子矩阵
  for (int i = 1; i <= 3; i++) {
    std::cout << "Block of size " << i << std::endl;
    std::cout << matrix4f.block(0, 0, i, i) << std::endl << std::endl;
  }
  // 1.2 block用于右值操作
  Eigen::Array44f m = Eigen::Array44f::Constant(0.6);
  matrix4f.block<2, 2>(1, 1) = m.block(1, 1, 2, 2);
  std::cout << "matrix4f is: " << std::endl;
  std::cout << matrix4f << std::endl << std::endl;

  /** 2. 行子式和列子式 **/
  std::cout << "2nd Row: " << matrix4f.row(1) << std::endl << std::endl;
  m.col(2) += 3 * m.col(3);
  std::cout << "After adding 3 times the first colunm into third colum, the "
               "matrix m is: \n";
  std::cout << matrix4f << std::endl << std::endl;

  /** 3. 边角子矩阵 **/
  std::cout << "leftCols(2) = " << std::endl << matrix4f.leftCols(2) << std::endl << std::endl;
  std::cout << "bottomRows<2>() = " << std::endl
            << matrix4f.bottomRows<2>() << std::endl
            << std::endl;

  matrix4f.topLeftCorner(1, 3) = matrix4f.bottomRightCorner(3, 1).transpose();
  std::cout << "After assignment, matrix4f = " << std::endl << matrix4f << std::endl;

  /** 4. 向量的子向量操作 **/
  Eigen::ArrayXf v(6);
  v << 1, 2, 3, 4, 5, 6;
  std::cout << "v.head(3) = " << std::endl << v.head(3) << std::endl << std::endl;
  std::cout << "v.tail<3>() = " << std::endl << v.tail<3>() << std::endl << std::endl;
  v.segment(1, 4) *= 2;
  std::cout << "after 'v.segment(1,4) *=  2', v = " << std::endl << v << std::endl;
  return 0;
}