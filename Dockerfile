FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install -y build-essential autoconf libtool pkg-config \
    wget \
    git \
    libssl-dev \
    libopencv-dev

RUN wget -q -O cmake-linux.sh https://github.com/Kitware/CMake/releases/download/v3.17.0/cmake-3.17.0-Linux-x86_64.sh
RUN sh cmake-linux.sh -- --skip-license
RUN rm cmake-linux.sh

WORKDIR /dependencies/
RUN git clone --recurse-submodules -b v1.33.2 https://github.com/grpc/grpc
WORKDIR /dependencies/grpc/
RUN mkdir build
WORKDIR /dependencies/grpc/build/
RUN cmake -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF ..
RUN make -j$(nproc)
RUN make install

COPY . /app/
WORKDIR /app/
RUN mkdir build
WORKDIR /app/build/
RUN cmake -DCMAKE_BUILD_TYPE=Release -G "CodeBlocks - Unix Makefiles" ..
RUN make -j$(nproc)
RUN make install

ENTRYPOINT ["grpc_va_server"]
