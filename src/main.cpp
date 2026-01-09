#include "../include/tensor.h"
#include <iostream>

int main()
{
    try
    {
        std::cout << "=== Test 1: Creating a Double Tensor ===" << std::endl;
        // 创建一个 3x4 的双精度张量
        naive::Tensor<double> t1({3, 4});
        t1.print_info();

        std::cout << "\n=== Test 2: Creating a Float Tensor ===" << std::endl;
        // 创建一个 5x5 的单精度张量
        naive::Tensor<float> t2({5, 5});
        t2.print_info();

        // 修改数据测试：把第一个元素改成 3.14
        t2.data()[0] = 3.14f;
        std::cout << "First element of t2: " << t2.data()[0] << std::endl;

        std::cout << "\n=== Test 3: Fill Function ===" << std::endl;
        naive::Tensor<float> t3({2, 3}); // 创建一个 2x3 的张量
            std::cout << "Data at index 0: " << t3.data()[0] << std::endl;

        t3.fill(6.66f);                  // 全部填成 6.66
        t3.print_info();                 // 打印形状

        std::cout << "Data at index 0: " << t3.data()[0] << std::endl;
        std::cout << "Data at index 5: " << t3.data()[5] << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}