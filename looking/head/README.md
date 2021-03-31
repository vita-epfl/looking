# Head model

## Training 

After getting the data, you can train the head model in this directory. The supported models are :
* ResNet18
* ResNet50
* AlexNet

You can run the training process using the command ```python main.py```.

### Arguments

Arguments: 

| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| -epochs --e     | 20 | Number of epochs |
| -learning_rate --lr     | 0.0001 | Learning Rate |
| -split --s  | original | Dataset split described in the paper (pedestrian, video, original) |
| -model --m     | resnet50 | Model type [resnet18, resnet50, alexnet] |
| -path --pt   | "./models" | path for model saving |
| -save          | False | Save the model |

For example, if you want to train a resnet50 using the original split and save the resulting model, just run :

```
python main.py --s original --m resnet50 -save
```

## Evaluation

