# Head + Joints

***Which model worked better*** ? The late fusion model on pretrained Resnet18 worked the best.
![alt text](https://github.com/younesbelkada/looking_model/blob/main/head%2Bjoints/Screenshot%20from%202021-03-03%2019-48-41.png)

## Training 

After getting the data, you can train the head model in this directory. The supported models are :
* ResNet18
* ResNet50

### Arguments

Arguments: 

| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| -epochs --e     | 20 | Number of epochs |
| -learning_rate --lr     | 0.0001 | Learning Rate |
| -split --s  | original | Dataset split described in the paper (video, original) |
| -model --m     | resnet18 | Model type [resnet18, resnet50] |
| -path --pt   | "./models" | path for model saving |
| -save          | False | Save the model |

### Example

If you want to train the model with a resnet50 backbone on the video based split, run :
```python main_test.py -m resnet50 -s video```
Make sure that you have created a directory called ```models``` in order to save your models and evaluate them later


## Evaluation

***After the training is done*** run the script with the exact same flags used for the training. This will output the evaluation metrics on Kitti and JAAD (with the 10 times sampling enabled)

### Arguments

Arguments: 

| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| -split --s  | original | Dataset split described in the paper (video, original) |
| -model --m     | resnet18 | Model type [resnet18, resnet50] |

### Example

If you want to train the model trained before (resnet50 on the video split) run :
```python inf_test.py -m resnet50 -s video```



