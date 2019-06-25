#!/usr/bin/env bash
path=`pwd`
nvidia-docker run -it --rm --net host --volume ${path}:/app:rw -e "PYTHONPATH=/app:$PYTHONPATH" atlasnet2
