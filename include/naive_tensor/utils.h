#ifndef NAIVE_UTILS_H
#define NAIVE_UTILS_H

#include "tensor.h"
#include <random>
#include <ctime>
#include <cmath>

namespace naive {
namespace utils {

    // ------------------------------------------------
    // 生成 1D 泊松矩阵 (N x N)
    // A = tridiag(-1, 2, -1)
    // ------------------------------------------------
    template <typename T>
    Tensor<T> generate_poisson_1d(int n) {
        Tensor<T> A({n, n});
        A.fill(static_cast<T>(0)); // 先全填 0 (HPC 关键：防止野值)

        T* data = A.data();

        // 填充三对角线
        for (int i = 0; i < n; ++i) {
            // 1. 主对角线 (2.0)
            // 索引映射: row=i, col=i -> index = i*n + i
            data[i * n + i] = static_cast<T>(2.0);

            // 2. 左下对角线 (-1.0)
            if (i > 0) {
                data[i * n + (i - 1)] = static_cast<T>(-1.0);
            }

            // 3. 右上对角线 (-1.0)
            if (i < n - 1) {
                data[i * n + (i + 1)] = static_cast<T>(-1.0);
            }
        }
        return A;
    }

    // ------------------------------------------------
    // 生成随机向量 (长度 N)
    // 用于造 b
    // ------------------------------------------------
    template <typename T>
    Tensor<T> generate_random_vector(int n) {
        Tensor<T> v({n});
        T* data = v.data();

        // 使用 C++ 标准库的随机数生成器
        std::mt19937 gen(time(nullptr)); 
        std::uniform_real_distribution<double> dis(-1.0, 1.0);

        for (int i = 0; i < n; ++i) {
            data[i] = static_cast<T>(dis(gen));
        }
        return v;
    }

} // namespace utils
} // namespace naive

#endif