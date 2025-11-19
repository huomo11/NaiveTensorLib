#include <iostream>
#include "Matrix.h" // 引入你自己写的头文件

// 暂时不写 CUDA 核函数，先测试 CPU 逻辑

int main()
{
    std::cout << "=== CPU Matrix Test Start ===" << std::endl;

    // 1. 创建两个矩阵 (3x3)
    Matrix<float> A(3, 3);
    Matrix<float> B(3, 3);

    // 2. 填充数据
    A.fill(1.0f); // A 全是 1
    B.fill(2.0f); // B 全是 2

    // 3. 打印看看
    std::cout << "Matrix A:" << std::endl;
    A.print();
    std::cout << "Matrix B:" << std::endl;
    B.print();

    // 4. 执行 CPU 矩阵乘法
    std::cout << "Calculating C = A * B on CPU..." << std::endl;
    Matrix<float> C = A * B;

    // 5. 验证结果 (应该是 3x3 的矩阵，每个元素是 1*2 + 1*2 + 1*2 = 6)
    std::cout << "Result Matrix C:" << std::endl;
    C.print();

    std::cout << "=== CPU Matrix Test End ===" << std::endl;
    return 0;
}