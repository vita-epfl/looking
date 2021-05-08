#!/bin/sh
BASE_PATH=$(pwd)
PATH_OUT="/home/younesbelkada/Travail/data"
ID_JR='1hazKAEdWQjjwPwmksL8XbyT9W8A6zn0Z'
PATH_SPLITS="/home/younesbelkada/Travail/looking/looking_new/create_data/splits_jack"

echo "Downloading JackRabbot dataset"

mkdir -p "${PATH_OUT}/JackRabbot"

#gdown --id $ID_JR
#mv reviewed.zip "${PATH_OUT}/JackRabbot"

#cd "${PATH_OUT}/JackRabbot"
#unzip reviewed.zip

cd $BASE_PATH

python create_nu_jr.py --train_file trainval.txt --test_file test.txt --path_txt $PATH_SPLITS --path_out "${PATH_OUT}/JackRabbot"