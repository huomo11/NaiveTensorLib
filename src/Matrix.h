#pragma once
#include <vector>
#include <iostream>

template <typename T>
// Matrix类的声明
class Matrix
{
private:
    // 私有属性
    int rows;
    int cols;

    // 核心：一维数组模拟二维矩阵
    std::vector<T> data;

public:
    // 公共函数

    // 构造矩阵
    Matrix(int r, int c);

    // 访问数据
    T &at(int i, int j);
    const T &at(int i, int j) const;

    // 打印矩阵
    void print();

    // 填充矩阵
    void fill(T val);

    // 矩阵乘法
    Matrix<T> operator*(const Matrix<T> &other);
};

// 1. 实现构造函数
template <typename T>
Matrix<T>::Matrix(int r, int c)
{
    // 记录形状
    rows = r;
    cols = c;

    // 分配空间
    data.resize(rows * cols);
}

// 2. 实现访问函数
template <typename T>
T &Matrix<T>::at(int i, int j)
{
    // 安全检查函数（暂未实现）

    int index = i * cols + j;

    return data[index];
}

template <typename T>
const T &Matrix<T>::at(int i, int j) const
{
    int index = i * cols + j;
    return data[index];
}

// 3. 实现打印函数
template <typename T>
void Matrix<T>::print()
{
    std::cout << "Matrix (" << rows << "x" << cols << "):" << std::endl;
    for (int i{0}; i < rows; ++i)
    {
        for (int j{0}; j < cols; ++j)
        {
            std::cout << at(i, j) << " ";
        }
        // 每一行打完都换行
        std::cout << std::endl;
    }
}

// 4. 实现填充矩阵单一元素
template <typename T>
void Matrix<T>::fill(T val)
{
    // for (int i{ 0 }; i < rows; ++i)
    //{
    //	for (int j{ 0 }; j < cols; ++j)
    //		at(i, j) = val;
    // }
    for (int k{0}; k < data.size(); ++k)
    {
        data[k] = val;
    }
}

// 5. 实现矩阵乘法
template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T> &other)
{
    if (cols != other.rows)
    {
        std::cerr << "维度不匹配" << std::endl;
    }

    Matrix<T> result(rows, other.cols);

    // 核心计算
    for (int i{0}; i < rows; ++i)
    {
        for (int j{0}; j < other.cols; ++j)
        {
            T sum = 0;
            for (int k{0}; k < cols; ++k)
            {
                sum += this->at(i, k) * other.at(k, j);
            }
            result.at(i, j) = sum;
        }
    }
    return result;
}
