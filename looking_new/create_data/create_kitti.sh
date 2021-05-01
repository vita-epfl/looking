#!/bin/sh
BASE_PATH=$(pwd)
PATH_OUTPUT_DATA="/home/younesbelkada/Travail/data/Kitti"
ID_ANNO='1kjmWOdyNP-us8SCNoPq6fRWTToli4FNo'
ID_IM='1Lr_RAkpAswnoYF7ZcYoLstMsTFiMy8Pm'

echo "Downloading Kitti annotations ..."
#gdown --id $ID_ANNO 

mv "kitti_annotation_2k30.tar.xz" "$PATH_OUTPUT_DATA"
cd "$PATH_OUTPUT_DATA"
tar -xvf kitti_annotation_2k30.tar.xz
mv kitti_new kitti_annotation

echo "Downloading Kitti Images ..."
#gdown --id $ID_IM
tar -xvf Kitti_images.tar.xz


