# process_look_data

## Define your own split

In order to create your data, you need first to download the annotation file ***that contains the images***. (E.g. the annotations that are shared in [this gdrive folder](https://drive.google.com/drive/folders/1KFFQPFfcCtCUp5pgOLqofhP7P6ElQhpV?usp=sharing). Each scene needs to have a folder, and inside this folder a folder called ```anno_<folder_name>``` containing the json annotations.

Then, create 2 txt files, for e.g. ```train.txt``` and ```test.txt``` and in each line of this file, add the absolute path to each scene that you want to train/test on. An example is given in this repo.

## Prepare before running the script

Create 2 directories ```nu``` and ```jack```. 

## Run the script

run ```python create_data.py <train_file> <test_file> <dataset_type>``` the last variable can be either ```nu``` or ```jack```.

## Understanding the output

This script outputs:
* Every single joint instance, in a ```.json``` format
* The cropped head images 
* A ```.txt``` file where at every line the instance is mapped to its original image + label.

## The txt file

The structure of a line in the txt ```txt``` is as follows:
```<absolute_path_to_the_original_image>, <split(train or test)>, <name_of_the_instance_file>, <x1_bb>, <y1_bb>, <x2_bb>, <y2_bb>, <looking>```
each instance file (instance file = the file that contains the joints for a specific instance) has its associated cropped head image. The naming is exactly the same, it's ```<name_of_the_instance_file>.png```

All the instances should be stored either in the ```nu``` or ```jack``` file created before

## BBox tuning

You need to tune the enlarging function for each dataset manually, please refer to the line ```30``` in the script ```create_data.py``` in order to modify it for each dataset

## Process new data

Everytime a new scene has been annotated, add the aboslute path of the scene folder in the corresponding txt file and the script will take care of the rest

## Example script

```python create_data.py trainvalJack.txt testJack.txt jack```

# How can we train / evaluate etc:

You will just need to create a Dataloader class. An example is given in the ```dataset_example.py``` please refer to this code. The code is adapted for a head+joints configuration. If you want only the head or only the joints, modify the script in a way that in ```preprocess``` you do not consider the heads (or the joints) and in ```__get_item()__``` you do not load the ```head```.

An example of a usage case of this class is implemented in the ```test.py``` file.
