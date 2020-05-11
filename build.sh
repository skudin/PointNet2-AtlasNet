#!/usr/bin/env bash
current_uid=`id -u`
current_gid=`id -g`
nvidia-docker build -t atlasnet2 --build-arg uid=${current_uid} --build-arg gid=${current_gid} -f Dockerfile .
