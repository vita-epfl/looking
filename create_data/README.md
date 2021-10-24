# Dataset creation

Please follow carefully the instructions below to configure and build the datasets

## 1. Create the LOOK dataset

### 0.1. Get the ground truth annotation file

Get directly the ground truth annotation file from [here](https://drive.google.com/file/d/1JUQgtIhXiPc-hAv7Fr6eu3No66VpyayM/view?usp=sharing) and download it in the current directory.

### 1.1. Getting the raw images

#### 1.1.1. Fast download

You can download all the images directly from this [link](https://drive.google.com/file/d/1OfsIIY9ljhCkZfLqL8n_n--X8oNBPfpj/view?usp=sharing).

#### 1.1.2. Manual download

Follow the instructions provided at the [benchmark website](https://looking-epfl.github.io/)

### 1.2. Getting the keypoints

After extracting the raw images, run ```run_pifpaf.py``` with the following arguments

| Parameter                 |Description   |
| :------------------------ |:-------------|
| ```--path_data```  | Path to the LOOK dataset |
| ```--path_out``` | Path to the output folder |
| ```--check``` | Checkpoint to use for Pifpaf, check the [official repo](https://openpifpaf.github.io/intro.html) for more details |
| ```--instance-threshold``` | Instance threshold parameter for OpenPifPaf |

This may take some time depending on your machine configuration, running this script on a GPU is recommanded.

### 1.3. Building the LOOK dataset

After extracting the raw images and downloading the ground truth file, run ```create_look.py``` with the following arguments:

| Parameter                 |Description   |
| :------------------------ |:-------------|
| ```--path_gt```  | Path to the LOOK ground truth annotation file |
| ```--path_out_txt``` | Path to the output text file. Default ```./splits_look```. Preferably not to change. |
| ```--path_output_files``` | Path to the output files where the keypoints and crops will be stored  |
| ```--path_keypoints``` | Path to the keypoints file obtained with PifPaf |
| ```--path_images``` | Path to the LOOK raw images |


## 2.Create the JAAD dataset 

First clone [JAAD repo](https://github.com/ykotseruba/JAAD) in your machine and download the JAAD dataset. In order to download JAAD dataset, clone the repo above, run ```sh download_clips.sh``` and ```sh split_clips_to_frames.sh```.

### 2.0 Easy Download and building

You can run ```sh create_jaad.sh``` after changing the variables name in the file. This script will automate the whole process, even the keypoints will be downloaded directly from the web. Check the explanation of the variables below:

| Parameter                 |Description   |
| :------------------------ |:-------------|
| ```PATH_OUTPUT_DATA```  | Path to the output files where the keypoints and crops will be stored  |
| ```PATH_JAAD_GT_OUT``` | Path to the output text file. Default ```./splits_look```. Preferably not to change. |
| ```PATH_JAAD``` | Path to the JAAD official repo  |
| ```ID_ANNO``` | Not to change |


### 2.1. Get the keypoints

#### 2.1.1 If you want to download our keypoints: 

The keypoints from pifpaf [here](https://drive.google.com/file/d/1zUAUALzhOuaaqYcogZdmpq6jTqSst6eL/view) and extract them in your machine. A folder named ```output_pifpaf_01_2k30``` should be outputted after the extraction.

#### 2.1.2 If you want to run Pifpaf keypoints detector by yourself

Run ```run_pifpaf.py``` with the following arguments

| Parameter                 |Description   |
| :------------------------ |:-------------|
| ```--path_data```  | Path to the JAAD images |
| ```--path_out``` | Path to the output folder |
| ```--check``` | Checkpoint to use for Pifpaf, check the [official repo](https://openpifpaf.github.io/intro.html) for more details |
| ```--instance-threshold'``` | Instance threshold parameter for OpenPifPaf |

* Run ```python3 dataset.py --path [path_to_JAAD_repo] --path_joints [path_to_extracted_joints_folder] --dir_out [path_to_the_directory_to_store_the_images_and_the_keypoints]```. This process can take up to an hour
* Run ```python3 split.py --gt_name [name_of_gt_txt_file] --txt_out [folder_of_txt_files]``` to create the splits

#### 2.1.3 If you want to run another keypoints detector by yourself

Make sure that you have at least one prediction file per image and the output folder should have the same structure as the input folders from JAAD as the following with the extension ```FILENAME.predictions.json```.

```
JAAD/images/
│
└─── video_0001/
│   │ 
│   | 00000.png.predictions.json
│   │ 00001.png.predictions.json
│   │   .
│   │   . 
│   │   .
└─── video_0002/
│   │ ...
│   
│ ...   
│   
└─── video_0003/
│   │ ...
```
Each prediction file, should be exactly the same as the output format of OpenPifPaf i.e, should contain an array of dictionaries for each detected instance as the following:
```
[{"keypoints": [247.85, 199.75, 0.64, 249.16, 198.17, 0.64, 246.71, 198.37, 0.57, 253.14, 198.59, 0.66, 245.92, 199.56, 0.01, 257.42, 206.13, 0.81, 
244.91, 207.36, 0.81, 264.34, 214.2, 0.88, 243.18, 216.53, 0.7, 257.19, 217.9, 0.7, 241.7, 224.71, 0.55, 256.65, 226.43, 0.78, 247.88, 227.21, 0.77,
 259.86, 239.61, 0.73, 248.46, 241.22, 0.74, 263.93, 248.97, 0.72, 250.95, 253.22, 0.75], "bbox": [239.43, 197.25, 27.67, 59.17], "score": 0.716, "c
ategory_id": 1}, {"keypoints": [915.47, 162.62, 0.48, 916.12, 160.59, 0.01, 913.52, 160.42, 0.57, 912.09, 161.98, 0.01, 908.12, 161.36, 0.71, 909.26
, 172.31, 0.82, 902.03, 171.29, 0.92, 911.96, 187.71, 0.71, 895.83, 186.05, 0.86, 923.07, 195.92, 0.69, 889.16, 200.43, 0.81, 906.45, 201.24, 0.76, 
903.73, 201.57, 0.85, 900.94, 223.12, 0.77, 915.82, 223.25, 0.76, 891.29, 240.47, 0.7, 925.68, 249.86, 0.73], "bbox": [885.54, 158.91, 45.29, 96.09]
, "score": 0.712, "category_id": 1}]
```

Each dictionary, should contain 3 values ```keypoints```, ```score``` and ```bbox```. 
* ```keypoints```: Is an array of 51 values, containing the x,y coordinate and the confidence score for each joint according to COCO format.
* ```bbox```: Is an array containing the bounding box ```x1, y1, w, h``` of the keypoints.
* ```score```: Confidence score of the predicted instance.

### 2.2. Build the JAAD dataset

Once we have the keypoints and the raw images, we need to build the dataset. Run the following script to build it:

```
python dataset.py --path $PATH_JAAD --path_joints ${PATH_KEYPOINTS} --dir_out $PATH_OUTPUT_DATA
python split.py --txt_out $PATH_JAAD_GT_OUT
```

| Parameter                 |Description   |
| :------------------------ |:-------------|
| ```--path```  | Path to the JAAD official repository |
| ```--path_joints``` | Path to the keypoints files |
| ```--dir_out``` | Path to the output files where the keypoints and crops will be stored  |
| ```--txt_out``` | Path to the output text file. Default: ```./split_jaad```. Preferably not to change |

Alternatively you can run ```sh create_jaad.sh``` after changing the variables name in the file. This script will automate the process, even the keypoints will be downloaded directly from the web.

## 3.1. Create the PIE dataset