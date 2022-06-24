#!/bin/bash


WORKING_DIR=$(pwd)

LOCAL_WORKING_DIR=${WORKING_DIR}/working


for folder in $(ls ${LOCAL_WORKING_DIR}/)
do
    echo "Processing ${folder}"
    python3 centroid-tracking.py \
                --dat_file_path ${LOCAL_WORKING_DIR}/${folder}/ \
                --frames ${LOCAL_WORKING_DIR}/${folder}/frames \
                --method cg_points \
                --output_path working_centroid
    echo "Done."
done
