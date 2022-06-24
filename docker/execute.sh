#!/bin/bash

export INPUT_VIDEO="${HOME}/data/chute.avi"
export OUTPUT_PATH="${HOME}/working"
export THRESH=0.9
export CUDA_VISIBLE_DEVICES='0'
export USER_ID=`cat /openpose/USER_ID.txt`

function print_help(){
    echo "Usage: openpose.sh [options]"
    echo ""
    echo "options:"
    echo "-h, --help                Show brief help"
    echo "--input_video             Path to input video"
    echo "--output_path             Path to output directory"
    echo "--thresh                  Threshould used to filtering out bad segments (default=${THRESH})"
    echo "--gpu                     Device ID"
}

if [ $# -eq 0 ]
then
    print_help
    exit 0
fi

while test $# -gt 0; do
  case "$1" in
    -h|--help)
        print_help
        exit 0
        ;;
    --input_video)
        export INPUT_VIDEO="$2"
        shift
        shift
        ;;
    --output_path)
        export OUTPUT_PATH="$2"
        shift
        shift
        ;;
    --thresh)
        export THRESH="$2"
        shift
        shift
        ;;
    --gpu)
        export CUDA_VISIBLE_DEVICES="$2"
        shift
        shift
        ;;
    *)
        echo -e "\nInvalid Parameter ($1)\n"
        exit 0
        ;;
  esac
done

CURRENT_DIR=$(pwd)
BASE_NAME=$(basename ${INPUT_VIDEO})
FILE_NAME=${BASE_NAME%.*}


OUTPUT_DIR=${OUTPUT_PATH}/${FILE_NAME}
mkdir -p ${OUTPUT_DIR}/frames
mkdir -p ${OUTPUT_DIR}/jsons
mkdir -p ${OUTPUT_DIR}/detections


# -- extract the frames
chown -R ${USER_ID}:${USER_ID} ${OUTPUT_PATH}
ffmpeg -i ${INPUT_VIDEO} "${OUTPUT_DIR}/frames/%09d.png"


# -- run the openpose.bin
./build/examples/openpose/openpose.bin --image_dir ${OUTPUT_DIR}/frames --display 0 --write_json ${OUTPUT_DIR}/jsons --write_images ${OUTPUT_DIR}/detections

# -- create videos with detections
ffmpeg -r:v 30 -i ${OUTPUT_DIR}/detections/%09d_rendered.png -codec:v libx264 -pix_fmt yuv420p -an ${OUTPUT_DIR}/poset_${BASE_NAME} -y
ffmpeg -i ${OUTPUT_DIR}/poset_${BASE_NAME} ${OUTPUT_DIR}/poset_${FILE_NAME}.avi

# -- convert detections to DVideo format
# cp json2dvideow.py ${OUTPUT_DIR}/jsons
# cd ${OUTPUT_DIR}/jsons
python3 /openpose/json2dvideow.py ${OUTPUT_DIR}/jsons
# cd ${CURRENT_DIR}

chown -R ${USER_ID}:${USER_ID} ${OUTPUT_PATH}

