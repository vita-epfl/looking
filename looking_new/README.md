# Are you looking at the camera ? Official repository

[Main image](https://raw.githubusercontent.com/vita-epfl/looking/younes/looking_new/images/people-walking-on-pedestrian-lane-during-daytime.pedictions.png?token=ALXVUF3F4QYCBPP7RXYD74TAS7RNS)

## Requirements

Use ```python>=3.6.9```. Run ```pip3 install -r requirements.txt```

## Create the datasets

Please follow the instructions on the folder ``` create_data/```.

## Train your models

You have one config file to modify. **Do not change the variables name**

**General parameters for training**:

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

**Model parameters** :

| Parameter                 |Description   |
| :------------------------ |:-------------|
| ```type```  | Choice of poses to train with between [```joints```, ```heads```, ```heads+joints```] |
| ```backbone``` | Backbone model for heads model. Applicable only if [```heads```, ```heads+joints```] selected above |
| ```fine_tune``` | Enable finetuning. Applicable only if [```heads```, ```heads+joints```] selected above |
| ```trained_on``` | Applicable only if ```heads+joints``` selected. Must be the name of the dataset where we trained the heads model. |

**Dataset parameters** : This section has to be modified only if you train your model on a single-dataset configuration. For a multi dataset configuration the parameters in the ```Multi_dataset``` parameters will be considered.

| Parameter                 |Description   |
| :------------------------ |:-------------|
| ```name```  | Name of the dataset to train on, choice between [```JAAD```, ```Kitti```, ```Nu```, ```JackRabbot```] |
| ```path_data``` | Path of the dataset folder of the selected dataset. Please refer to ```create_data/``` folder |
| ```split``` | Splitting strategy, applicable only if [```JAAD````] selected above. Choice between [```scenes```, ```instances```] |
| ```path_txt``` | path to the ground truth txt file, this parameter shouldn't be modified if the dataset has been created correctly |



## Evaluate your trained models

## Visualizing some results

