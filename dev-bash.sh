#!/bin/bash
# Open a shell inside the pipeline image, with ./data and ./output mounted
# the same way as the README's run command. Build the image first:
#   docker build -t nipype-preprocess-on-brain-mr .
IMAGE="${IMAGE:-nipype-preprocess-on-brain-mr}"
docker run --rm -it \
    --memory 8g \
    --user "$(id -u):$(id -g)" \
    --volume "$PWD/data:/data" \
    --volume "$PWD/output:/output" \
    "$IMAGE" bash
