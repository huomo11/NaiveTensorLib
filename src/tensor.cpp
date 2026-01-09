#include "../include/naive_tensor/tensor.h"
#include <numeric>
#include <algorithm>

namespace naive
{

    // 构造函数
    template <typename T>
    Tensor<T>::Tensor(const std::vector<int> &shape) : shape_(shape)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("Shape cannot be empty");
        }

        // 计算总大小
        size_ = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());

        // 申请堆内存
        data_ = new T[size_];

        // 初始化为0
        std::fill(data_, data_ + size_, static_cast<T>(0));
    }

    // 析构函数
    template <typename T>
    Tensor<T>::~Tensor()
    {
        if (data_ != nullptr)
        {
            delete[] data_;
            data_ = nullptr;
        }
    }

    // 打印信息
    template <typename T>
    void Tensor<T>::print_info() const
    {
        std::cout << "Tensor Shape :[";
        for (size_t i = 0; i < shape_.size(); ++i)
        {
            std::cout << shape_[i] << (i < shape_.size() - 1 ? ", " : "");
        }
        std::cout << "]" << std::endl;
    }

    // 填充函数
    template <typename T>
    void Tensor<T>::fill(T value)
    {
        for (size_t i = 0; i < size_; i++)
        {
            data_[i] = value;
        }
        
    }

    template class Tensor<float>;
    template class Tensor<double>;

}