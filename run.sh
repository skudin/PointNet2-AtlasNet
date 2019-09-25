#!/usr/bin/env bash
path=`pwd`
nvidia-docker run -it --rm --net host --shm-size=8G --volume ${path}:/app:rw atlasnet2
