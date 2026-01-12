cmake_minimum_required(VERSION 3.10)
project(NaiveTensorLib)

# 1. 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. [关键修正] 设置头文件搜索路径
# 你的头文件都在 include/naive_tensor 下
# 添加这个路径后，main.cpp 中的 #include "tensor.h" 才能找到文件
include_directories(${CMAKE_SOURCE_DIR}/include/naive_tensor)

# (可选) 添加 include 根目录，方便以后使用 #include "naive_tensor/tensor.h" 写法
include_directories(${CMAKE_SOURCE_DIR}/include)

# 3. 指定源文件
# 你的 main.cpp 在 src 目录下
set(SOURCES 
    src/main.cpp 
)

# 4. 生成可执行文件
add_executable(run ${SOURCES})

# 5. 编译优化
if(MSVC)
    target_compile_options(run PRIVATE /O2)
else()
    target_compile_options(run PRIVATE -O3 -march=native)
endif()