cmake_minimum_required(VERSION 3.15...3.29)
project(deepx-excuter-cuda LANGUAGES CXX CUDA) 

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# 设置编译优化
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
set(CMAKE_BUILD_TYPE Debug)


include(FetchContent)
FetchContent_Declare(
    cutlass
    GIT_REPOSITORY git@github.com:NVIDIA/cutlass.git
    GIT_TAG v3.8
)
FetchContent_MakeAvailable(cutlass)

# 包含头文件目录
include_directories(src)
include_directories(${cutlass_SOURCE_DIR}/include)
include_directories(${cutlass_SOURCE_DIR}/tools/util/include)


add_subdirectory(../cpp-common common)  # 包含公共模块

# 源文件
file(GLOB_RECURSE DEEPX_SOURCES "src/deepx/*.cpp" "src/deepx/*.cu")
file(GLOB_RECURSE CLIENT_SOURCES "src/client/*.cpp")

include(FetchContent)

find_package(CUDAToolkit REQUIRED COMPONENTS cublas)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

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
    cutlass
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
