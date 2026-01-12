#include "naive_tensor/tensor.h"
#include "naive_tensor/utils.h"
#include "naive_tensor/ops.h"
#include "naive_tensor/cg.h"
#include <iostream>
#include <vector>

using namespace naive;

int main() {
    // ---------------------------------------------------------
    // 1. 设置问题规模
    // ---------------------------------------------------------
    int N = 1000;
    std::cout << "Setting up Poisson 1D problem, N = " << N << std::endl;

    // ---------------------------------------------------------
    // 2. 生成数据
    // ---------------------------------------------------------
    // 生成 A (Poisson 1D Matrix, SPD)
    Tensor<double> A = utils::generate_poisson_1d<double>(N);
    
    // 生成随机 b
    Tensor<double> b = utils::generate_random_vector<double>(N);
    
    // 初始解 x0 (全0)
    Tensor<double> x({N});
    
    // 简单的 Jacobi 预处理 (Poisson 1D 对角线元素全是 2.0)
    Tensor<double> inv_diag({N});
    inv_diag.fill(0.5); 

    // ---------------------------------------------------------
    // 3. 运行 AMP-PCG 求解器
    // ---------------------------------------------------------
    solver::AmpParams params;
    params.tol = 1e-10;
    // params.max_iter = 2000;  <-- 删除这一行，因为 max_iter 已经在下面的函数参数里了
    
    // 调整阈值以观察切换效果 (因为 Poisson 问题收敛很快)
    params.tau_z_s = 1e-1; // 相对残差 < 0.1 时切 FP32
    params.tau_z_h = 1e-4; // 相对残差 < 1e-4 时切 FP16

    // max_iter (2000) 作为第 5 个参数传递
    solver::amp_pcg_solve(A, b, inv_diag, x, 2000, params);

    // ---------------------------------------------------------
    // 4. 验证结果
    // ---------------------------------------------------------
    // Check True Residual: ||b - Ax||
    Tensor<double> Ax({N});
    ops::gemv(A, x, Ax); // Ax = A * x
    
    Tensor<double> r_final = b;
    // r = b - Ax (using axpy: r = -1.0 * Ax + r)
    ops::axpy(-1.0, Ax, r_final); 
    
    double final_norm = ops::norm(r_final);
    std::cout << "\nFinal True Residual Norm ||b - Ax|| = " << final_norm << std::endl;

    return 0;
}