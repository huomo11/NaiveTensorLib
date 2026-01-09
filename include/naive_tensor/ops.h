#ifndef NAIVE_OPS_H
#define NAIVE_OPS_H

#include "tensor.h"
#include <cassert> // 用来检查两个向量长度是否一样

namespace naive {
namespace ops {

    // ------------------------------------------------
    // 任务 1: 实现 AXPY (y = alpha * x + y)
    // ------------------------------------------------
    template <typename T>
    void axpy(const Tensor<T>& x, Tensor<T>& y, T alpha) {
        // 安全检查：长度必须一样
        assert(x.size() == y.size());

        const T* x_ptr = x.data(); // 拿到 x 的钥匙（只读）
        T* y_ptr = y.data();       // 拿到 y 的钥匙（可写）
        int n = x.size();

        // [你的任务]：写一个 for 循环
        // 逻辑：y_ptr[i] 变成 alpha * x_ptr[i] + y_ptr[i]
        for (int i = 0; i < n; ++i) {
            y_ptr[i] += alpha * x_ptr[i];
        }
    }

    // ------------------------------------------------
    // 任务 2: 实现 Dot Product (res = x * y)
    // ------------------------------------------------
    template <typename T>
    double dot(const Tensor<T>& x, const Tensor<T>& y) {
        assert(x.size() == y.size());
        
        const T* x_ptr = x.data();
        const T* y_ptr = y.data();
        int n = x.size();

        // [HPC 关键点]：累加器必须用 double，防止精度溢出！
        // 哪怕 T 是 float，也要用 double 存 sum。
        double sum = 0.0;

        // [你的任务]：写一个 for 循环累加
        for (int i = 0; i < n; ++i) {
            sum += x_ptr[i] * y_ptr[i];
        }

        return sum;
    }

    // ------------------------------------------------
    // 任务 3: 实现 Copy (y = x)
    // ------------------------------------------------
    template <typename T>
    void copy(const Tensor<T>& x, Tensor<T>& y) {
        assert(x.size() == y.size());
        
        const T* x_ptr = x.data();
        T* y_ptr = y.data();
        int n = x.size();

        // [你的任务]：写一个 for 循环，把 x 搬到 y
        for (int i = 0; i < n; ++i) {
            y_ptr[i] = x_ptr[i];
        }
    }

} // namespace ops
} // namespace naive

#endif // NAIVE_OPS_H