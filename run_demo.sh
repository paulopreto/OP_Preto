#!/bin/bash

WORKING_DIR=$(pwd)
DOCKER_NAME=openpose
CUDA_VISIBLE_DEVICES=0

LOCAL_DATASET_DIR=${WORKING_DIR}/data
LOCAL_WORKING_DIR=${WORKING_DIR}/working

DOCKER_DATASET_DIR=/dataset
DOCKER_WORKING_DIR=/working


for fname in $(ls ${LOCAL_DATASET_DIR}/*.*)
do
    filename=$(basename ${fname})
    nvidia-docker run \
        -it --rm --userns=host --pid=host \
        -v ${HOME}:${HOME} \
        -v ${LOCAL_DATASET_DIR}:${DOCKER_DATASET_DIR} \
        -v ${LOCAL_WORKING_DIR}:${DOCKER_WORKING_DIR} \
        ${DOCKER_NAME} --input_video ${DOCKER_DATASET_DIR}/${filename} --output_path ${DOCKER_WORKING_DIR} --gpu ${CUDA_VISIBLE_DEVICES}
done

