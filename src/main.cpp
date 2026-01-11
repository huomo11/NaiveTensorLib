#include "naive_tensor/tensor.h"
#include "naive_tensor/ops.h"
#include "naive_tensor/cg.h"    // 确保这里包含了你的 pcg_solve
#include "naive_tensor/utils.h" // 包含泊松矩阵生成器
#include <iostream>
#include <iomanip> // 用于 std::scientific

int main() {
    int N = 100; // 矩阵规模
    std::cout << "=== Test Ground: 1D Poisson Equation (N=" << N << ") ===" << std::endl;

    // ---------------------------------------------------------
    // 1. 准备数据
    // ---------------------------------------------------------
    
    // 生成 A (100x100 泊松矩阵)
    auto A = naive::utils::generate_poisson_1d<double>(N);

    // 设定真解 x_true (全是 1.0)
    naive::Tensor<double> x_true({N});
    x_true.fill(1.0);

    // 计算 b = A * x_true
    naive::Tensor<double> b({N});
    naive::ops::gemv(A, x_true, b);

    // 初始猜测 x (全 0)
    naive::Tensor<double> x({N});
    x.fill(0.0);

    // [新增关键点] 准备 Jacobi 预处理器 (对角线逆矩阵)
    // 1D 泊松矩阵的主对角线全是 2.0，所以逆矩阵对角线全是 1/2 = 0.5
    naive::Tensor<double> inv_diag({N});
    inv_diag.fill(0.5);

    // ---------------------------------------------------------
    // 2. 求解 (调用 PCG)
    // ---------------------------------------------------------
    std::cout << "Starting PCG solver (with Jacobi)..." << std::endl;
    
    // [修复点] 参数顺序: A, b, inv_diag, x, max_iter, tol
    naive::solver::pcg_solve(A, b, inv_diag, x, N * 2, 1e-10);

    // ---------------------------------------------------------
    // 3. 验证结果
    // ---------------------------------------------------------
    
    // 计算误差向量: error = x - x_true
    // 逻辑: x = 1.0 * x + (-1.0 * x_true)
    
    // [适配] 使用你的新接口: scal(alpha, x)
    naive::ops::scal(-1.0, x_true); 
    
    // [适配] 使用你的新接口: axpy(alpha, x, y) -> y = alpha*x + y
    // 这里把变负的 x_true 加到 x 上
    naive::ops::axpy(1.0, x_true, x); 

    // 计算误差模长
    double error_norm = naive::ops::norm(x);

    std::cout << "=== Verification ===" << std::endl;
    std::cout << "L2 Error Norm: " << std::scientific << error_norm << std::endl;

    if (error_norm < 1e-6) {
        std::cout << "SUCCESS! Solver is accurate." << std::endl;
    } else {
        std::cout << "WARNING! Large error detected." << std::endl;
    }

    return 0;
}