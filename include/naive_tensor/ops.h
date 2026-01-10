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
        // ------------------------------------------------
        template <typename T>
        void axpy(const Tensor<T> &x, Tensor<T> &y, T alpha)
        {
            assert(x.size() == y.size());

            const T *x_ptr = x.data();
            T *y_ptr = y.data();
            int n = x.size();

            for (int i = 0; i < n; ++i)
            {
                y_ptr[i] += alpha * x_ptr[i];
            }
        }

        // ------------------------------------------------
        // Dot Product: res = x * y
        // ------------------------------------------------
        template <typename T>
        double dot(const Tensor<T> &x, const Tensor<T> &y)
        {
            assert(x.size() == y.size());

            const T *x_ptr = x.data();
            const T *y_ptr = y.data();
            int n = x.size();

            double sum = 0.0; // Use double to prevent overflow

            for (int i = 0; i < n; ++i)
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
            int n = x.size();

            for (int i = 0; i < n; ++i)
            {
                y_ptr[i] = x_ptr[i];
            }
        }

        // ------------------------------------------------
        // GEMV: y = A * x
        // A is (M, N), x is (N), y is (M)
        // ------------------------------------------------
        template <typename T>
        void gemv(const Tensor<T> &A, const Tensor<T> &x, Tensor<T> &y)
        {
            const T *A_ptr = A.data();
            const T *x_ptr = x.data();
            T *y_ptr = y.data();

            int M = y.size(); 
            int N = x.size(); 

            // Row-Major Layout: A_ij is at A_ptr[i * N + j]
            for (int i = 0; i < M; ++i)
            {
                double sum = 0.0;
                for (int j = 0; j < N; ++j)
                {
                    sum += A_ptr[i * N + j] * x_ptr[j];
                }
                y_ptr[i] = static_cast<T>(sum);
            }
        }

        // ------------------------------------------------
        // L2 Norm: sqrt(x * x)
        // ------------------------------------------------
        template <typename T>
        double norm(const Tensor<T> &x)
        {
            return std::sqrt(dot(x, x));
        }

        // ------------------------------------------------
        // Scal: x = alpha * x
        // ------------------------------------------------
        template <typename T>
        void scal(Tensor<T> &x, T alpha)
        {
            T *x_ptr = x.data();
            int n = x.size();

            for (int i = 0; i < n; ++i)
            {
                x_ptr[i] *= alpha;
            }
        }

    } // namespace ops
} // namespace naive

#endif // NAIVE_OPS_H