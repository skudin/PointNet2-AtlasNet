# Base dist.
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

MAINTAINER Stepan Kudin <kudin.stepan@yandex.ru>

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
 && rm -rf /var/lib/apt/lists/*


# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it.
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user
