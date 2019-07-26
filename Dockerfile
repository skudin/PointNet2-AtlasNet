# Base dist.
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

MAINTAINER Stepan Kudin <kudin.stepan@yandex.ru>

ENV PYTHONPATH=/app:$PYTHONPATH

# Install some basic utilities
RUN apt-get update --fix-missing && apt-get install -y \
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
    python3-minimal \
    python3-pip \
    gcc \
 && rm -rf /var/lib/apt/lists/*

# Pytorch and torchvision.
RUN pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl && \
    pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl

# Install requirements.
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# Install additional packages.
COPY packages/chamfer /tmp/chamfer
RUN cd /tmp/chamfer && \
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
