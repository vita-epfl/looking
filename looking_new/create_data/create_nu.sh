#!/bin/sh
BASE_PATH=$(pwd)
PATH_OUT="/home/younesbelkada/Travail/data"
ID_TRAIN='1U2nrSUFNhJHy6zqJBiqp0h4eyo3pu50c'
ID_TEST='1tjYoPbE-4rvsMF8qW6FqjGQNTBPcTi_y'
PATH_SPLITS="/home/younesbelkada/Travail/looking/looking_new/create_data/splits_nu"

echo "Downloading Nuscenes dataset"

mkdir -p "${PATH_OUT}/Nuscenes"

cd "${PATH_OUT}/Nuscenes"

#gdown --id $ID_TRAIN
#unzip trainset.zip

#gdown --id $ID_TEST
#unzip testset.zip

cd $BASE_PATH

python create_nu_jr.py --train_file trainval.txt --test_file test.txt --path_txt $PATH_SPLITS --path_out "${PATH_OUT}/Nuscenes" --name 'nu'