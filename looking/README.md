# looking_model

The repository for the looking model 

## Requirements

```pip install -r requirements.txt```

## Preprocessing

### JAAD
* clone the JAAD repository somewhere else. And copy the file ```generate_csv.py``` there. Change the variable ```jaad_path``` to your path to the JAAD repository. Then run ```generate_csv.py```. After that copy the file ```results_new.csv``` in the main folder here.
* Copy the annotations from pifpaf here : [ici](https://drive.google.com/file/d/1zUAUALzhOuaaqYcogZdmpq6jTqSst6eL/view) 
* In the file ```create_data.py``` modify the lines corresponding to the paths to the annotations and to JAAD dataset images accroding to yours. Run the command `python create_data.py` to obtain a txt with location and label of all the pedestrians in JAAD dataset. This script will generate a file called ```ground_truth_2k30.txt```.
* To split the instances, download the splits from   
`https://github.com/ykotseruba/JAAD/tree/JAAD_2.0/split_ids/default` and save them in the `splits` directory.
* Inside the ```splits/``` directory, run `python split_jaad_video.py` & `python split_jaad_original.py` 
to create the training, val, test split accordingly to the instance-based split and the scene-based one.

**How the csv file works ??**
Each line is an instance from the gt of JAAD dataset, the structure is as follows


<video_name>, <pedestrian_code>, <frame_number>, <bounding_box_y1>, <bounding_box_x1>, <bounding_box_y2>, <bounding_box_x2>, <looking_label>

### KITTI
For Kitti Dataset :

Training : 2011_09_29/2011_09_29_drive_0071_sync/image_03/data

Testing : data_object_image_2/training/image_2

* Create a directory called ```Kitti``` inside your ```./data/``` directory that already contains JAAD dataset.
* Create 2 directories : ```Trainset``` and ```Testset``` somewhere in the computer to store the images and the annotations
* Copy all the images of the trainset on this folder and same for testset
* Download the annotations here (1.4MB) : [here](https://drive.google.com/file/d/1kjmWOdyNP-us8SCNoPq6fRWTToli4FNo/view?usp=sharing)
* Copy the folders ```Testset_2k30``` and ```Trainset_2k30``` respectively inside ```Trainset``` and ```Testset```
* Modify the ```create_data_kitti.py``` : line 31 modify it with the path to the annotations ```anno_Trainset```. 
Line 32 : modify it with the path to the images on the training set ```Trainset```. Same for lines 63 & 64 and the testset.
* Run ```create_data_kitti.py```

At the end of the script tou should have a file called ```kitti_gt.txt``` 
and the folder ```./data/Kitti/``` should contain ```.png``` images and ```.json``` files.

### NuScenes
Training set & test set available togther [here](https://drive.google.com/drive/folders/1KFFQPFfcCtCUp5pgOLqofhP7P6ElQhpV?usp=sharing) the images are shared as well.

the path to the nuscenes images : ```Asia/samples/```



### JACKRABBOT

Please refer to the ```pedestrians``` folder for pedestrians counting.

For the annotations : get the data [here](https://drive.google.com/drive/folders/1KFFQPFfcCtCUp5pgOLqofhP7P6ElQhpV?usp=sharing)

Here the images are provided together with the annotations

### PIE
get the data [here](https://drive.google.com/file/d/1kjmWOdyNP-us8SCNoPq6fRWTToli4FNo/view?usp=sharing)

Here only the **matched** joints are provided together with their gt label. There is no track about their path, image name etc. I would recommand you to do from scratch for PIE.

## Training the head, joints or heads+joints model
Please refer to the ```head, joints, or head+ joints``` directory 


### Multi-dataset training

* **How to train on JAAd and Nuscenes and evaluate on PIE for example?**
for now supported only for JAAD and Kitti. If you want to use the deprecated dataset you can follow the instructions on this repository : [here](https://github.com/younesbelkada/looking_multi)

## Ablation study
* **How can I run the study of the table about body joints, head joints etc?**
Please refer to the ```joints``` folder


## Annotations
for processing the annotations, please refer to [this](https://github.com/younesbelkada/process_look_data) private repo (the invitation has been sent on Friday)

