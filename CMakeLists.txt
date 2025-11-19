cmake_minimum_required(VERSION 3.10)

# 声明项目语言：C++ (CXX) 和 CUDA
project(NaiveTensorLib LANGUAGES CXX CUDA)

# 强制使用 C++17 (现代 HPC 标配)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 设置 CUDA 架构 (针对你的 RTX 3050 优化)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# 查找源码 (包含 .cpp 和 .cu)
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")

# 生成可执行文件
add_executable(main ${SOURCES})

# 开启分离编译 (允许在不同文件中调用 CUDA 函数)
set_target_properties(main PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

message(STATUS "Setup complete. Ready to build.")