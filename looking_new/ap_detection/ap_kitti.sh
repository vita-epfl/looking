#!/bin/sh
BASE_PATH=$(pwd)
ID_PIF=100KhQD-C7HANHz9VPj-zgIYaEHLgRb68
ID_GT=181gsyO_YkFrANkH1GJPpz-V_W09HRQhO
PATH_OUT="/home/younesbelkada/Travail/data/AP_out"
PATH_KITTI="/home/younesbelkada/Travail/data/Kitti/Kitti_images/Test"

mkdir -p $PATH_OUT
cd $PATH_OUT

echo "Downloading Kitti gt annotations ..."
gdown --id $ID_GT

unzip ground_truth.zip
mv label_2 ground_truth

echo "Downloading Kitti predicted labels ..."
gdown --id $ID_PIF

unzip predictions.zip
mv test predictions

cd $BASE_PATH

python ap_kitti.py --path "${PATH_OUT}/ground_truth" --path_pred $PATH_KITTI --path_pif "${PATH_OUT}/predictions"
python main.py -q -np -d Kitti