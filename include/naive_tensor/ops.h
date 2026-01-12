#ifndef NAIVE_OPS_H
#define NAIVE_OPS_H

#include "tensor.h"
#include <cassert>
#include <cmath>

namespace naive
{
    namespace ops
    {

        // ------------------------------------------------
        // AXPY: y = alpha * x + y
        // Level 1 BLAS
        // ------------------------------------------------
        template <typename T>
        void axpy(T alpha, const Tensor<T> &x, Tensor<T> &y)
        {
            assert(x.size() == y.size());

            const T *x_ptr = x.data();
            T *y_ptr = y.data();
            size_t n = x.size();

            for (size_t i = 0; i < n; ++i)
            {
                y_ptr[i] += alpha * x_ptr[i];
            }
        }

        // ------------------------------------------------
        // Dot Product: res = x^T * y
        // Level 1 BLAS
        // ------------------------------------------------
        template <typename T>
        double dot(const Tensor<T> &x, const Tensor<T> &y)
        {
            assert(x.size() == y.size());

            const T *x_ptr = x.data();
            const T *y_ptr = y.data();
            size_t n = x.size();

            double sum = 0.0; // 使用 double 累加防止溢出与精度损失

            for (size_t i = 0; i < n; ++i)
            {
                sum += x_ptr[i] * y_ptr[i];
            }

            return sum;
        }

        // ------------------------------------------------
        // Copy: y = x
        // ------------------------------------------------
        template <typename T>
        void copy(const Tensor<T> &x, Tensor<T> &y)
        {
            assert(x.size() == y.size());

            const T *x_ptr = x.data();
            T *y_ptr = y.data();
            size_t n = x.size();

            for (size_t i = 0; i < n; ++i)
            {
                y_ptr[i] = x_ptr[i];
            }
        }

        // ------------------------------------------------
        // Scal: x = alpha * x
        // Level 1 BLAS
        // ------------------------------------------------
        template <typename T>
        void scal(T alpha, Tensor<T> &x)
        {
            T *x_ptr = x.data();
            size_t n = x.size();

            for (size_t i = 0; i < n; ++i)
            {
                x_ptr[i] *= alpha;
            }
        }

        // ------------------------------------------------
        // L2 Norm: ||x||_2 = sqrt(x^T * x)
        // ------------------------------------------------
        template <typename T>
        double norm(const Tensor<T> &x)
        {
            return std::sqrt(dot(x, x));
        }

        // ------------------------------------------------
        // Element-wise Multiplication: z = x .* y
        // ------------------------------------------------
        template <typename T>
        void elwise_mult(const Tensor<T> &x, const Tensor<T> &y, Tensor<T> &z)
        {
            assert(x.size() == y.size());
            assert(y.size() == z.size());

            const T *x_ptr = x.data();
            const T *y_ptr = y.data();
            T *z_ptr = z.data();
            size_t n = x.size();

            for (size_t i = 0; i < n; ++i)
            {
                z_ptr[i] = x_ptr[i] * y_ptr[i];
            }
        }

        // ------------------------------------------------
        // GEMV: y = A * x
        // A: (M, N) Row-Major
        // x: (N)
        // y: (M)
        // Level 2 BLAS
        // ------------------------------------------------
        template <typename T>
        void gemv(const Tensor<T> &A, const Tensor<T> &x, Tensor<T> &y)
        {
            size_t M = y.size();
            size_t N = x.size();

            assert(A.size() == M * N);

            const T *A_ptr = A.data();
            const T *x_ptr = x.data();
            T *y_ptr = y.data();

            for (size_t i = 0; i < M; ++i)
            {
                double sum = 0.0;
                for (size_t j = 0; j < N; ++j)
                {
                    sum += A_ptr[i * N + j] * x_ptr[j];
                }
                y_ptr[i] = static_cast<T>(sum);
            }
        }

        // [新增] 模拟 FP16 下溢 (Underflow)
        // 论文核心机制：当数值小于 FP16 最小正规数 (约 6.1e-5) 时，强制归零
        template <typename T>
        void quantize_fp16(Tensor<T> &x)
        {
            T *ptr = x.data();
            int n = x.size();

            // FP16 minimum normal number ~ 6.1e-5
            const double FP16_MIN = 6.1e-5;

            for (int i = 0; i < n; ++i)
            {
                // 绝对值太小，直接截断为 0 (模拟 Underflow)
                if (std::abs(ptr[i]) > 0 && std::abs(ptr[i]) < FP16_MIN)
                {
                    ptr[i] = static_cast<T>(0.0);
                }
            }
        }

        // [新增] 模拟 FP32 (Single Precision)
        // 作用：模拟单精度的精度损失 (截断尾数)
        template <typename T>
        void quantize_fp32(Tensor<T> &x)
        {
            T *ptr = x.data();
            int n = x.size();
            for (int i = 0; i < n; ++i)
            {
                // 核心逻辑：转成 float 丢弃精度，再转回 T
                float temp = static_cast<float>(ptr[i]);
                ptr[i] = static_cast<T>(temp);
            }
        }

    } // namespace ops
} // namespace naive

#endif // NAIVE_OPS_H