cmake_minimum_required(VERSION 3.15...3.29)
project(deepx-excuter-ompsimd LANGUAGES CXX ) 

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# 设置编译优化
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
set(CMAKE_BUILD_TYPE Debug)
# 设置 SIMD 编译选项
 
# 包含头文件目录
include_directories(src)
 
# 源文件
file(GLOB_RECURSE SOURCES "src/*.cpp")

# cpu 线性代数库
list(APPEND CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/openblas-pthread/cmake")
find_package(OpenBLAS REQUIRED)
# 
# include(FetchContent)

# # Fetch OpenBLAS from GitHub
# FetchContent_Declare(
#   OpenBLAS
#   GIT_REPOSITORY https://github.com/OpenMathLib/OpenBLAS
#   GIT_TAG        v0.3.29  # 使用最新的稳定版本或您需要的特定版本
# )

# # 下载并构建 OpenBLAS
# FetchContent_MakeAvailable(OpenBLAS)

# 线程并行
find_package(OpenMP REQUIRED)
# simd
find_package(hwy REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -msse4.2")
# 内存池
find_package(PkgConfig REQUIRED)
pkg_check_modules(JEMALLOC REQUIRED jemalloc)


find_package(yaml-cpp REQUIRED)


find_package(Boost REQUIRED COMPONENTS system)
find_package(yaml-cpp REQUIRED)

# 包含Boost头文件路径
include_directories(${Boost_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME}
    PUBLIC 
    yaml-cpp
    ${JEMALLOC_LIBRARIES}    
    openblas
    # OpenBLAS
    OpenMP::OpenMP_CXX
    hwy
)


# 测试
# add_subdirectory(test/tensor)
# add_subdirectory(test/op)
# # 数据集测试
# add_subdirectory(test/dataset)
