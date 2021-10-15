#!/bin/sh
BASE_PATH=$(pwd)
PATH_JAAD_REPO="/home/younesbelkada/Travail/JAAD"
PATH_JAAD_TEXT_FILE="/home/younesbelkada/Travail/looking/looking_new/create_data/splits_jaad"

python ap_jaad.py --path $PATH_JAAD_REPO --txt_out $PATH_JAAD_TEXT_FILE
python main.py -d JAAD -q -np