FROM ubuntu:18.04

ENV DEBIAN_FRONTEND="noninteractive"

# Setup image with essentials
RUN apt-get update
RUN apt-get install -y build-essential git curl wget \
    apt-transport-https ca-certificates software-properties-common gnupg \
    autoconf libtool pkg-config libssl-dev \
    libopencv-dev lcov

# Install git lfs for downloading model files
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install -y git-lfs
RUN git lfs install

# Install cmake-3.13.5
WORKDIR /root
RUN wget http://www.cmake.org/files/v3.18/cmake-3.18.6.tar.gz
RUN tar xzf cmake-3.18.6.tar.gz && cd cmake-3.18.6 && ./configure && make -j $(nproc) && make install

# Install grpc and libonnxruntime from tool scripts
WORKDIR /root
COPY tools/install_grpc.sh /root/install_grpc.sh
RUN bash install_grpc.sh

COPY tools/install_onnxruntime.sh /root/install_onnxruntime.sh
RUN bash install_onnxruntime.sh
