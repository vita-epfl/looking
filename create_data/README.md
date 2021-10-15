# Dataset creation

Since we are running multi-modal training, the multi model dataset is a bit tricky to create. Please follow carefully the instructions below to configure and build the datasets

## Create the JAAD dataset 

First clone [JAAD repo](https://github.com/ykotseruba/JAAD) in your machine and download the JAAD dataset. In order to download JAAD dataset, clone the repo above, run ```sh download_clips.sh``` and ```sh split_clips_to_frames.sh```.

### If you want to download our keypoints: 

The keypoints from pifpaf [here](https://drive.google.com/file/d/1zUAUALzhOuaaqYcogZdmpq6jTqSst6eL/view) and extract them in your machine. A folder named ```output_pifpaf_01_2k30``` should be outputted after the extraction.

### If you want to run Pifpaf keypoints detector by yourself

Run ```run_pifpaf.py``` with the following arguments

| Parameter                 |Description   |
| :------------------------ |:-------------|
| ```--path_data```  | Path to the JAAD images |
| ```--path_out``` | Path to the output folder |
| ```--check``` | Checkpoint to use for Pifpaf, check the [official repo](https://openpifpaf.github.io/intro.html) for more details |
| ```--instance-threshold'``` | Instance threshold parameter for OpenPifPaf |

* Run ```python3 dataset.py --path [path_to_JAAD_repo] --path_joints [path_to_extracted_joints_folder] --dir_out [path_to_the_directory_to_store_the_images_and_the_keypoints]```. This process can take up to an hour
* Run ```python3 split.py --gt_name [name_of_gt_txt_file] --txt_out [folder_of_txt_files]``` to create the splits

### If you want to run another keypoints detector by yourself

Make sure that you have at least one prediction file per image 