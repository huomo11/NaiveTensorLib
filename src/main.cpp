#include "naive_tensor/tensor.h"
#include "naive_tensor/ops.h"
#include "naive_tensor/cg.h" // 引入刚才写的求解器
#include <iostream>

int main() {
    // 1. 准备数据
    // 构造 A (2x2)
    naive::Tensor<double> A({2, 2});
    double* A_ptr = A.data();
    A_ptr[0] = 2.0;  A_ptr[1] = -1.0;
    A_ptr[2] = -1.0; A_ptr[3] = 2.0;

    // 构造 b (长度2, 全是 1.0)
    naive::Tensor<double> b({2});
    b.fill(1.0);

    // 构造 x (长度2, 初始为 0.0)
    naive::Tensor<double> x({2});
    x.fill(0.0);

    // 2. 调用求解器
    std::cout << "=== Start CG Solver ===" << std::endl;
    naive::solver::cg_solve(A, b, x, 100, 1e-6);

    // 3. 验证结果
    std::cout << "=== Solution ===" << std::endl;
    std::cout << "x[0] = " << x.data()[0] << " (Expected: 1.0)" << std::endl;
    std::cout << "x[1] = " << x.data()[1] << " (Expected: 1.0)" << std::endl;

    return 0;
}