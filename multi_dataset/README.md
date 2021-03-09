# looking_multi (Deprecated)
Code for multi dataset - Looking model

## Get the dataset

Get the dataset from here and copy the json files in the ``` data/ ``` directory from [here](https://drive.google.com/drive/folders/1Vkxhe82B2avw74WoVWjcd_13yUfz-oXZ?usp=sharing) 

## Train and evaluate

| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| --tr_pr  | kitti,jack,jaad,pie | Dataset to use for training |
| --ts_pr   | kitti,jack,jaad,pie | Dataset to use for training |

If you wand for example to train your model on JAAD and PIE  and evaluate on Kitti, JR and Nuscenes run :

```python main.py --tr_pr jaad,pie --ts_pr jack,kitti,nu```

## Supported datasets

* JAAD
* Kitti
* PIE
* Nuscenes
* Jackrabbot
