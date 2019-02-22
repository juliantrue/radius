#!/bin/bash

# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

TENSORRT_TAR_FILE=$1
if [[ -z "${TENSORRT_TAR_FILE}" ]]; then
    echo "First argument is missing."
    echo "Usage  : build_redtail_image.sh <full_path_to_TensorRT_tar_file>"
    echo "Example: build_redtail_image.sh /data/downloads/NVIDIA/TensorRT-4.0.1.6.Ubuntu-16.04.4.x86_64-gnu.cuda-9.0.cudnn7.1.tar.gz"
    exit 1
fi

IMAGE_TAG_SUFFIX="-v2"
echo "Using ${IMAGE_TAG_SUFFIX} image suffix."

# Copy the file to Docker context first.
cp ${TENSORRT_TAR_FILE} .
cp ${CUDNN_TAR_FILE} .

# Build the image.
docker build -t redtail-sim:kinetic${IMAGE_TAG_SUFFIX} \
  --build-arg TENSORRT_TAR_FILE=`basename ${TENSORRT_TAR_FILE}` \
  -f Dockerfile.kinetic .
