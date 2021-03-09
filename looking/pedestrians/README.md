# How to calculate the pedestrian number on JR 

First download the gt labels of JR directly from [here](https://drive.google.com/drive/folders/1EAI2rTKsZ-Ht-caZ4qoNj8XNMw6Ql3QJ?usp=sharing). Extract the annotations and save them in ```./labels_2d``` directory.

Each label file has the following structure : <```SCENE_NAME```+```'_0_'```+```CAMERA_NAME```+```.json```>. Add in the ```file_names``` variable the relevant scene name that you have labeled. Then run :

```python calculate.py```

