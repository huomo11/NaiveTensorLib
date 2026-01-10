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
        // 构造函数
        Tensor(const std::vector<int> &shape);

        // 析构函数
        ~Tensor();

        // 拷贝构造函数
        Tensor(const Tensor &other);

        // 赋值运算符
        Tensor &operator=(const Tensor &other);

        // 获取原始数据指针
        T *data() { return data_; }
        const T *data() const { return data_; }

        // 获取属性
        int size() const { return size_; }
        const std::vector<int> &shape() const { return shape_; }

        // 工具函数
        void print_info() const;
        void fill(T value);

    private:
        std::vector<int> shape_;
        int size_;
        T *data_; // 指向堆内存的指针
    };

} // namespace naive

#endif // NAIVE_TENSOR_H