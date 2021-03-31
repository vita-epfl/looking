# Joints model

## Training 

After getting the data, you can train the joints model in this directory.
You can run the training process using the command ```python main.py```.

### Arguments

Arguments: 

| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| -e --epochs     | 30 | Number of epochs |
| -lr --lr     | 0.0001 | Learning Rate |
| -s --split  | original | Dataset split described in the paper (pedestrian, video, original) |
| -kt -kt | False | Test on Kitti dataset after training |
| -dr --dropout     | 0.2 | Dropout rate |
| -pt --path   | "./models" | path for model saving |
| -p --pose          | full | Which pose type to train [full, body, head] |

For example, if you want to train the model using full joints on the scenes split and evaluate on kitti after that just run :

```
python main.py --s video --p full -kt
```

The script will automatically test the model on the JAAD test set after the script.


## Predictor

### Arguments 

Arguments: 

| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| -image --im | "test.jpg" | image path |
| -out --o | "./out_images/" | out path |
| -model --m   | "./models/" | path for model |
| -add_kps --kps   | False | Add keypoints in the image for visualization |
| -add_gt --gt   | False | Compare with the gt annotation |
| -gt_file --gt_f   | "./gt_pred/" | path for the annotation file |
| -pose --p          | full | Which pose type to train [full, body, head] |

***Head / body  predictions not supported yet***

For example, if you want to predict the image called '000433.jpg', compare it with the ground truth and visualize the keypoints
```
python predict.py --m models/looking_model_jaad_video_new_full_kps.p --im 000433.jpg --kp --gt --gt_f gt_pred/000433.jpg.json
```
<p align="center">
  <img align="center" src="out_images/000433.prediction.png" width=80% height=80%>
</p>

