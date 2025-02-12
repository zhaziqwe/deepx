FROM docker.array2d.com/library/ubuntu:18.04

# 基础构建环境
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    libyaml-cpp-dev \
    libjemalloc-dev \
    clang \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 安装 Highway SIMD 库

ADD . .
RUN cd thirdlib/highway && \
    rm -rf build && mkdir build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j$(nproc) && make install && \
    cd ../../ && rm -rf thirdlib/highway

# 配置环境变量（参考代码块 `test/op/1_mem.cpp` startLine:16）
RUN rm -rf build && mkdir build && cd build && \
    cmake ..&& \
    make -j$(nproc)

CMD ["./build/bin/deepx-excuter-ompsimd"]