#!/bin/sh

## Do not modify

ID_KEYPOINTS='1L1ChkEw9PQb1eEub3InkvHlgkdQzbdjn'
ID_ANNOTATION='1SmT-D5E3_CCbAj7BUtvb5GOF2CNsNLhD'
BASE_PATH=$(pwd)
PATH_GT_OUT="./splits_look"
PATH_GT_FILE="./annotations_LOOK.csv"

## To Modify
# Path to the donwloaded LOOK dataset
PATH_IMAGES="LOOK_dataset"

# Aboslute path to the processed dataset
PATH_OUTPUT_DATA="output_images"

echo "Downloading LOOK annotations ..."
gdown --id $ID_ANNOTATION

mkdir -p $PATH_OUTPUT_DATA
cd $PATH_OUTPUT_DATA


echo "Downloading LOOK keypoints ..."
gdown --id $ID_KEYPOINTS
tar -xvf LOOK_keypoints.tar.gz

cd $BASE_PATH
echo "Building the LOOK dataset ..."
python3 create_look.py --path_gt $PATH_GT_FILE --path_out_txt $PATH_GT_OUT --path_output_files $PATH_OUTPUT_DATA --path_keypoints "${PATH_OUTPUT_DATA}/LOOK_keypoints" --path_images "${PATH_IMAGES}/LOOK"
