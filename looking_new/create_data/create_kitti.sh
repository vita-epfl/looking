#!/bin/sh
BASE_PATH=$(pwd)
PATH_OUTPUT_DATA="./Kitti"
PATH_KITTI_GT_OUT="./splits_kitti"
ID_ANNO='1kjmWOdyNP-us8SCNoPq6fRWTToli4FNo'
ID_IM='1Lr_RAkpAswnoYF7ZcYoLstMsTFiMy8Pm'

mkdir -p $PATH_OUTPUT_DATA
mkdir -p $PATH_KITTI_GT_OUT

echo "Downloading Kitti annotations ..."
gdown --id $ID_ANNO

mv "kitti_annotation_2k30.tar.xz" "$PATH_OUTPUT_DATA"
cd "$PATH_OUTPUT_DATA"
tar -xf kitti_annotation_2k30.tar.xz
mv kitti_new Kitti_annotations

echo "Downloading Kitti Images ..."
gdown --id $ID_IM
tar -xf Kitti_images.tar.xz

cd "$BASE_PATH"
python dataset.py --dataset Kitti --path_kitti "${PATH_OUTPUT_DATA}/Kitti_images" --path_kitti_anno "${PATH_OUTPUT_DATA}/Kitti_annotations" --txt_out $PATH_KITTI_GT_OUT --dir_out $PATH_OUTPUT_DATA
