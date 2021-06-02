FROM ubuntu:18.04

ENV DEBIAN_FRONTEND="noninteractive"

# Setup image with essentials
RUN apt-get update
RUN apt-get install -y build-essential git curl wget \
    apt-transport-https ca-certificates software-properties-common gnupg \
    autoconf libtool pkg-config \
    libopencv-dev lcov

# Install git lfs for downloading model files
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install -y git-lfs
RUN git lfs install

# Install cmake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(cat /etc/os-release | grep UBUNTU_CODENAME | sed 's/.*=//') main"
RUN apt-get update
RUN apt-get install cmake=3.15.0-0kitware1 cmake-data=3.15.0-0kitware1

# Install grpc and libonnxruntime from tool scripts
WORKDIR /root
COPY tools/install_grpc.sh /root/install_grpc.sh
RUN bash install_grpc.sh

COPY tools/install_onnxruntime.sh /root/install_onnxruntime.sh
RUN bash install_onnxruntime.sh
