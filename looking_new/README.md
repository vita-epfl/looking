# New repo

Simplified and efficient repo for looking

## Requirements

Use ```python>=3.6.9```. Run ```pip3 install -r requirements.txt```

## Create the datasets

Please follow the instructions on the folder ``` create_data/```.

## Train your models

You have one config file to modify. **Do not change the variables name**

General parameters for training:

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
| ```multi_dataset``` | Enables the mutli dataset training configuration. Choice between  [```yes```, ```no```].|ls

## Evaluate your trained models

## Visualizing some results

