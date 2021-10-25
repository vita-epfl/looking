# Do pedestrians pay attention? Eye contact detection for autonomous driving - Official repository

![alt text](https://github.com/vita-epfl/looking/blob/main/images/people-walking-on-pedestrian-lane-during-daytime.pedictions.png)

Image taken from : https://jooinn.com/people-walking-on-pedestrian-lane-during-daytime.html . Results obtained with the model trained on JackRabbot, Nuscenes, JAAD and Kitti. The model file is available at ```models/predictor``` and can be reused for testing with the predictor. 

## Table of Contents
  * [Requirements](#requirements)
  * [Build the datasets](#build-the-datasets)
  * [Train your models](#train-your-models)
    + [General parameters for training](#--general-parameters-for-training---)
    + [Model parameters](#--model-parameters----)
    + [Dataset parameters](#--dataset-parameters----)
    + [Multi-Dataset parameters](#--multi-dataset-parameters---)
    + [LOOK-Dataset parameters](#--look-dataset-parameters---)
  * [Evaluate your trained models](#evaluate-your-trained-models)
    + [General parameters for evaluation](#general-parameters-for-evaluation)
    + [Evaluate on LOOK / Use a trained model on LOOK](#evaluate-on-look---use-a-trained-model-on-look)
    + [Evaluate on JAAD or PIE](#evaluate-on-jaad-or-pie)
  * [Predictor](#predictor)

## Requirements

Use ```python>=3.6.9```. Run ```pip3 install -r requirements.txt```

## Build the datasets

Please follow the instructions on the folder ``` create_data/```.

## Train your models

You have one config file to modify. **Do not change the variables name**

### General parameters for training:

| Parameter                 |Description   |
| :------------------------ |:-------------|
| ```epochs```  | Number of epochs for training |
| ```learning_rate``` | Learning rate for training |
| ```path``` | Path to save the models |
| ```batch_size``` | Batch size for training |
| ```pose``` | Choice of poses to train with between [```full```, ```body```, ```head```] |
| ```loss``` | Choice of loss function to use with between [```BCE```, ```Focal```] |
| ```dropout``` | Dropout rate |
| ```grad_map``` | Enables gradient maps visualization after training. Applicable only for ```joints``` models. Choice between  [```yes```, ```no```].|
| ```device``` | cuda device |
| ```optimizer``` | Choice of optimizers between [```adam```, ```sgd```] |
| ```eval_it``` | Number of iterations for negative sampling on the validation set |
| ```multi_dataset``` | Enables the mutli dataset training configuration. Choice between  [```yes```, ```no```].|

### Model parameters :

| Parameter                 |Description   |
| :------------------------ |:-------------|
| ```type```  | Choice of poses to train with between [```joints```, ```heads```, ```heads+joints```] |
| ```backbone``` | Backbone model for heads model. Applicable only if [```heads```, ```heads+joints```] selected above |
| ```fine_tune``` | Enable finetuning. Applicable only if [```heads```, ```heads+joints```] selected above |
| ```trained_on``` | Applicable only if ```heads+joints``` selected. Must be the name of the dataset where we trained the heads model. |

### Dataset parameters :

This section has to be modified only if you train your model on a single-dataset configuration. For a multi dataset configuration the parameters in the ```Multi_dataset``` parameters will be considered.

| Parameter                 |Description   |
| :------------------------ |:-------------|
| ```name```  | Name of the dataset to train on, choice between [```JAAD```, ```LOOK```, ```PIE```] |
| ```path_data``` | Path of the dataset folder of the selected dataset. Please refer to ```create_data/``` folder |
| ```split``` | Splitting strategy, applicable only if [```JAAD```] selected above. Choice between [```scenes```, ```instances```]. Otherwise you can put anything, it will be ignored. |
| ```path_txt``` | path to the ground truth txt files, this parameter shouldn't be modified if the dataset has been created correctly. Default: ```./create_data``` |

### Multi-Dataset parameters: 
| Parameter                 |Description   |
| :------------------------ |:-------------|
| ```train_datasets```  | Name of the datasets to train on separated by a comma **wihtout a space**. Examples: ```JAAD,PIE```, ```LOOK,JAAD```, ```LOOK,PIE,JAAD```] |
| ```weighted``` | Enable the weighted sampling while training. Choice between [```yes```, ```no```] |

### LOOK-Dataset parameters: 
| Parameter                 |Description   |
| :------------------------ |:-------------|
| ```data```  | Name of subset to train the model on. Choice between [```all```, ```Kitti```, ```JRDB```, ```Nuscenes```] |
| ```trained_on``` | Applicable only for evaluation |

After changing your configuration file, run:

```python train.py --file [PATH_TO_CONFIG_FILE]```

A sample config file can be found at ```config_example.ini```

## Evaluate your trained models

Here you need to specify 2 things. The dataset **you have trained your model** and the dataset **you want to evaluate your model**.

The dataset you have trained your model should be specified on the ```General```, ```Model_type``` and ```Dataset``` sections. If you have trained a model with some specific parameters (heads model, fine tuned, backbone, etc.) you should use the same parameters you have used to train your model.

### General parameters for evaluation

Take a look at the ```Eval``` section. Here are the details for each variable:
| Parameter                 |Description   |
| :------------------------ |:-------------|
| ```eval_on```  | Name of the datasets to evaluate. Choice between [```JAAD```, ```LOOK```, ```PIE```] |
| ```height``` | Enable the ablation study on the heights of the pedestrians (see the paper for more details). Choice between [```yes```, ```no```] |
| ```split``` | Splitting strategy, applicable only if [```JAAD```] selected above. Choice between [```scenes```, ```instances```]. Otherwise you can put anything, it will be ignored. |
| ```path_data_eval``` | Path where the built data is stored. |


### Evaluate on LOOK / Use a trained model on LOOK

If you have trained a model on LOOK and/or you want to evaluate your model on LOOK, you should modify the section ```LOOK```.
| Parameter                 |Description   |
| :------------------------ |:-------------|
| ```data```  | Name of subset to evaluate the model on. Choice between [```all```, ```Kitti```, ```JRDB```, ```Nuscenes```] |
| ```trained_on``` | Which subset the trained model has been trained. Choice between [```all```, ```Kitti```, ```JRDB```, ```Nuscenes```]  |

### Evaluate on JAAD or PIE

If you want to evaluate your model on JAAD or PIE, you should modify the ```JAAD_dataset``` or ```PIE_dataset``` section:
| Parameter                 |Description   |
| :------------------------ |:-------------|
| ```path_data```  | Path where the built data is stored |
| ```split``` | Splitting strategy, applicable only if [```JAAD```] selected above. Choice between [```scenes```, ```instances```]. Otherwise you can put anything, it will be ignored. |
| ```path_txt``` | path to the ground truth txt files, this parameter shouldn't be modified if the dataset has been created correctly. Default: ```./create_data``` |

## Predictor

![alt text](https://github.com/vita-epfl/looking/blob/main/images/kitti.gif =1238x375)

Get predictions from our pretrained model using any image with the predictor. You need to specify the following arguments:

| Parameter                 |Description   |
| :------------------------ |:-------------|
| ```--glob``` | Glob expression to be used. Example: ```.png``` |
| ```--images```  | Path to the input images. If glob is enabled you need the path to the directory where you have the query images |
