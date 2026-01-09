#include "../include/naive_tensor/tensor.h"
#include "../include/naive_tensor/ops.h" // 引入新写的算子
#include <iostream>

int main() {
    // 1. 造数据
    naive::Tensor<double> x({5});
    naive::Tensor<double> y({5});
    x.fill(1.0); // x 全是 1
    y.fill(2.0); // y 全是 2

    // 2. 测试 AXPY: y = 2.0 * x + y
    // 预期结果：y 变成 2*1 + 2 = 4
    std::cout << "Testing AXPY..." << std::endl;
    naive::ops::axpy(x, y, 2.0);
    
    // 打印 y 的第一个元素看看对不对
    std::cout << "y[0] (should be 4): " << y.data()[0] << std::endl;

    // 3. 测试 Dot: x * y
    // 预期结果：1 * 4 * 5个元素 = 20
    // 注意：现在的 y 已经是 4 了，x 还是 1
    std::cout << "Testing Dot..." << std::endl;
    double res = naive::ops::dot(x, y);
    std::cout << "Dot result (should be 20): " << res << std::endl;

    return 0;
}