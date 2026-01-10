#ifndef NAIVE_CG_H
#define NAIVE_CG_H

#include "tensor.h"
#include "ops.h"
#include <iostream>
#include <cmath>

namespace naive
{
    namespace solver
    {
        // ----------------------------------------------------------------
        // Conjugate Gradient Solver (CG)
        // 求解 Ax = b
        // A: 稀疏矩阵 (这里暂时用稠密矩阵代替)
        // b: 右端项向量
        // x: 解向量 (初始猜测值，也是输出结果)
        // ----------------------------------------------------------------
        template <typename T>
        void cg_solve(const Tensor<T> &A, const Tensor<T> &b, Tensor<T> &x, int max_iter, double tol)
        {
            // --- 1. 初始化 ---
            // 假设初始 x 为 0，则初始残差 r = b - A*0 = b
            Tensor<T> r = b;
            Tensor<T> p = b;
            Tensor<T> Ap(b.shape());

            double rho = ops::dot(r, r);
            double rho_old = rho;

            std::cout << "Initial Residual: " << std::sqrt(rho) << std::endl;

            // --- 2. 迭代求解 ---
            for (int k = 0; k < max_iter; ++k)
            {
                // 检查收敛性
                if (std::sqrt(rho) < tol)
                {
                    std::cout << "Converged at iter " << k << "!" << std::endl;
                    break;
                }

                // 1. Ap = A * p
                ops::gemv(A, p, Ap);

                // 2. alpha = (r, r) / (p, A * p)
                double pAp = ops::dot(p, Ap);
                double alpha = rho / pAp;

                // 3. x = x + alpha * p
                ops::axpy(p, x, alpha);

                // 4. r = r - alpha * Ap
                ops::axpy(Ap, r, -alpha);

                // 5. 更新 beta 和搜索方向 p
                double rho_new = ops::dot(r, r);
                double beta = rho_new / rho_old;

                // p = r + beta * p
                // 分两步实现：p = beta * p; p = p + r;
                ops::scal(p, static_cast<T>(beta));
                ops::axpy(r, p, static_cast<T>(1.0));

                // 更新 rho
                rho_old = rho_new;
                rho = rho_new;

                if (k % 10 == 0)
                {
                    std::cout << "Iter " << k << ", Residual: " << std::sqrt(rho) << std::endl;
                }
            }
        }

    } // namespace solver
} // namespace naive

#endif