# Base dist.
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

MAINTAINER Stepan Kudin <kudin.stepan@yandex.ru>

ENV DEBIAN_FRONTEND noninteractive
ENV PYTHONPATH=/app:$PYTHONPATH

# Install some basic utilities
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    apt-utils \
    build-essential \
    cmake \
    gdb \
    libeigen3-dev \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libglfw3-dev \
    libglu1-mesa-dev \
    libjsoncpp-dev \
    libosmesa6-dev \
    libpng-dev \
    lxde \
    mesa-utils \
    ne \
    python3-minimal \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-tk \
    python3-dbg \
    pybind11-dev \
    software-properties-common \
    x11vnc \
    xorg-dev \
    xterm \
    xvfb \
    curl \
    ca-certificates \
    sudo \
    git \
    wget \
    bzip2 \
    libx11-6 \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    unzip \
    gcc \
 && rm -rf /var/lib/apt/lists/*

# Pytorch and torchvision.
RUN pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl && \
    pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl

# Install requirements.
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

RUN mkdir /deps

# Install additional packages.
COPY packages/chamfer /deps/chamfer
RUN cd /deps/chamfer && \
    python3 setup.py build_ext --inplace && \
    pip3 install -e .

COPY packages/pointnet2_ext /deps/pointnet2_ext
RUN cd /deps/pointnet2_ext && \
    python3 setup.py build_ext --inplace && \
    pip3 install -e .

# Metro distance.
RUN mkdir /deps/metro
RUN cd /deps/metro && \
    git clone https://github.com/ThibaultGROUEIX/metro_sources.git . && \
    git checkout 792cf7ca797b77890f60d439ca9cb72999892e58 && \
    python3 setup.py --build

# Earth-Mover-Distance
RUN mkdir /deps/emd
RUN cd /deps/emd && \
    git clone https://github.com/daerduoCarey/PyTorchEMD.git . && \
    git checkout e5c0feb8c15741a858362f9c397d9ed4b8b1752b && \
    python3 setup.py build_ext --inplace && \
    pip3 install -e .

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it.
ARG uid=1000
ARG gid=1000
RUN addgroup --gid $gid user && \
    adduser --uid $uid --ingroup user --disabled-password --gecos '' --shell /bin/bash user && \
    chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

CMD [ "/bin/bash" ]
