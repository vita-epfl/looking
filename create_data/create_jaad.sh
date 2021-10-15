#!/bin/sh
BASE_PATH=$(pwd)
PATH_OUTPUT_DATA="/data/younes-data/JAAD"
PATH_JAAD_GT_OUT="./splits_jaad"
PATH_JAAD="/data/taylor-data/Datasets/JAAD"
ID_ANNO='1zUAUALzhOuaaqYcogZdmpq6jTqSst6eL'

mkdir -p $PATH_OUTPUT_DATA

cd $PATH_OUTPUT_DATA

echo "Downloading JAAD annotations ..."
gdown --id $ID_ANNO 

tar -xvf output_pifpaf_01_2k30.tar.gz

cd $BASE_PATH

python dataset.py --path $PATH_JAAD --path_joints "${PATH_OUTPUT_DATA}/output_pifpaf_01_2k30" --dir_out $PATH_OUTPUT_DATA

python split.py --txt_out $PATH_JAAD_GT_OUT