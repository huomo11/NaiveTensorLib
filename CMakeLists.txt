cmake_minimum_required(VERSION 3.10)

# 项目名称和版本
project(NaiveTensorLib VERSION 1.0 LANGUAGES CXX)

# --- 1. 基础设置 ---
# 使用 C++17 标准 (现代 C++ 开发的基础)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 告诉编译器头文件在哪里 (.h 文件)
include_directories(${CMAKE_SOURCE_DIR}/include)

# --- 2. 定义源代码 ---
# 这是一个名为 main_app 的可执行程序，它由 src/main.cpp 和 src/tensor.cpp 编译而来
add_executable(main_app 
    src/main.cpp 
    src/tensor.cpp
)

# --- 3. (预留) 测试模块 ---
# 目前先注释掉，等你写了测试代码再开启
# add_executable(run_tests tests/test_main.cpp)