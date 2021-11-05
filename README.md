# Do pedestrians pay attention? Eye contact detection for autonomous driving - Official repository

![alt text](https://github.com/vita-epfl/looking/blob/main/images/people-walking-on-pedestrian-lane-during-daytime.pedictions.png)

Image taken from : https://jooinn.com/people-walking-on-pedestrian-lane-during-daytime.html . Results obtained with the model trained on JackRabbot, Nuscenes, JAAD and Kitti. The model file is available at ```models/predictor``` and can be reused for testing with the predictor. 

## Table of contents

- [Requirements](#requirements)
- [Predictor](#predictor)
  * [Example command](#example-command-)
- [Create the datasets for training and evaluation](#create-the-datasets-for-training-and-evaluation)
- [Training your models on LOOK / JAAD / PIE](#training-your-models-on-look---jaad---pie)
- [Evaluate your trained models](#evaluate-your-trained-models)
- [Annotate new images](#annotate-new-images)


## Requirements

Use ```3.6.9 <= python < 3.9```. Run ```pip3 install -r requirements.txt``` to get the dependencies

## Predictor

<img src="https://github.com/vita-epfl/looking/blob/main/images/kitti.gif" data-canonical-src="https://github.com/vita-epfl/looking/blob/main/images/kitti.gif" width="1238" height="375" />

Get predictions from our pretrained model using any image with the predictor. The scripts extracts the human keypoints on the fly using [OpenPifPaf](https://openpifpaf.github.io/intro.html). The predictor supports eye contact detection using human keypoints only.
You need to specify the following arguments in order to run correctly the script:

| Parameter                 |Description   |
| :------------------------ |:-------------|
| ```--glob``` | Glob expression to be used. Example: ```.png``` |
| ```--images```  | Path to the input images. If glob is enabled you need the path to the directory where you have the query images |
| ```--looking_threshold```  | Threshold to define an eye contact. Default ```0.5```|
| ```--transparency```  | Transparency of the output poses. Default ```0.4```|

### Example command:

If you want to reproduce the result of the top image, run:

#### If you want to run the predictor on a GPU:
```
python predict.py --images images/people-walking-on-pedestrian-lane-during-daytime-3.jpg --looking_threshold 0.25
```
#### If you want to run the predictor on a CPU:
```
python predict.py --images images/people-walking-on-pedestrian-lane-during-daytime-3.jpg --looking_threshold 0.25 --device cpu --disable-cuda
```

## Create the datasets for training and evaluation

Please follow the instructions on the folder ``` create_data/```.

## Training your models on LOOK / JAAD / PIE

You have one config file to modify. **Do not change the variables name**. Check the meaning of each variable to change on the [training wiki](/wikis/train.md).

After changing your configuration file, run:

```python train.py --file [PATH_TO_CONFIG_FILE]```

A sample config file can be found at ```config_example.ini```

## Evaluate your trained models

Check the meaning of each variable to change on the [evaluation wiki](/wikis/eval.md).

After changing your configuration file, run:

```python evaluate.py --file [PATH_TO_CONFIG_FILE]```

A sample config file can be found at ```config_example.ini```

## Annotate new images

Check out the folder ```annotator``` in order to run our annotator to annotate new instances for the task.
