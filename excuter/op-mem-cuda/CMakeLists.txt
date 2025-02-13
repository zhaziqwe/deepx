cmake_minimum_required(VERSION 3.15...3.29)
project(deepx-excuter-cuda LANGUAGES CXX CUDA) 

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# 设置编译优化
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
set(CMAKE_BUILD_TYPE Debug)
# 设置 SIMD 编译选项
 
# 包含头文件目录
include_directories(src)
add_subdirectory(../common common)  # 包含公共模块

# 源文件
file(GLOB_RECURSE DEEPX_SOURCES "src/deepx/*.cpp" "src/deepx/*.cu")
file(GLOB_RECURSE CLIENT_SOURCES "src/client/*.cpp")

# include(FetchContent)

find_package(CUDAToolkit REQUIRED COMPONENTS cublas)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.6)  # 根据您的安装路径进行调整
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 61)  # 根据您的 GPU 计算能力进行调整
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON) #确保 CMake 能够正确识别 CUDA 文件并将其编译为目标


find_package(yaml-cpp REQUIRED)
 
add_library(deepx SHARED 
    ${DEEPX_SOURCES}
)

target_link_libraries( deepx
    PUBLIC 
    deepx_common
    yaml-cpp
    CUDA::cublas 
)
add_executable(${PROJECT_NAME} ${CLIENT_SOURCES})
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    deepx
)
# 测试
add_subdirectory(test/tensorfunc)
add_subdirectory(test/op)
# # 数据集测试
# add_subdirectory(test/dataset)
