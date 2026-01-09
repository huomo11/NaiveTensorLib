#ifndef NAIVE_TENSOR_H
#define NAIVE_TENSOR_H

#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>

namespace naive
{

    template <typename T>
    class Tensor
    {
    public:
        // 构造函数：创建一个shape形状的张量
        Tensor(const std::vector<int> &shape);

        // 析构函数：释放内存
        ~Tensor();

        // 获取裸数据指针
        T *data() { return data_; }
        const T *data() const { return data_; }

        // 获取元素总数
        int size() const { return size_; }

        // 打印张量信息
        void print_info() const;

        // 填充张量
        void fill(T value);

    private:
        std::vector<int> shape_;
        int size_;
        T *data_; // 指向堆内存的指针
    };

}

#endif // NAIVE_TENSOR_H