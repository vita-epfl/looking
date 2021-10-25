#!/bin/sh
ID_LOOK='1WH6fKrXW0AbbTkvOELinxHOfZPOFuLMv'
ID_KEYPOINTS='1L1ChkEw9PQb1eEub3InkvHlgkdQzbdjn'
BASE_PATH=$(pwd)
PATH_OUTPUT_DATA="~/data/LOOK/"
PATH_OUTPUT_DATA_ALL="~/data/LOOK/LOOK_all"
PATH_GT_OUT="./splits_look"
PATH_GT_FILE="./annotations.csv"

mkdir -p $PATH_OUTPUT_DATA
mkdir -p $PATH_OUTPUT_DATA_ALL

cd $PATH_OUTPUT_DATA
echo "Downloading LOOK images ..."
gdown --id $ID_LOOK
unzip LOOK_dataset.zip

echo "Downloading LOOK keypoints ..."
gdown --id $ID_KEYPOINTS
tar -xvf LOOK_keypoints.tar.gz

cd $BASE_PATH
echo "Building the LOOK dataset ..."
python create_look.py --path_gt $PATH_GT_FILE --path_out_txt $PATH_GT_OUT --path_output_files $PATH_OUTPUT_DATA_ALL --path_keypoints "${PATH_OUTPUT_DATA}/LOOK_keypoints" --path_images "${PATH_OUTPUT_DATA}/LOOK"
