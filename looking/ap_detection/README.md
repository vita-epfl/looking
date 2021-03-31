# Ap detection 

## JAAD test set

* First, run ```python split_jaad_video.py```
* ```mkdir input```
* ```mkdir input/ground-truth input/detection-results```
* ```python ap.py```
* ```python main.py```

## Kitti 

* Frist, download the 3 annotation folders from [here](https://drive.google.com/drive/folders/1i0W0eGUc6TllY2Dj1lU4BlcQHtW1tOV6?usp=sharing)
* ```mkdir input```
* ```mkdir input/ground-truth input/detection-results```
* modify the paths of the 3 variables between line 11-13 inside ```ap_kitti.py```. ```path_anno_kitti``` corresponds to the ```label_2``` folder (original kitti gt), ```path_anno_pifpaf``` corresponds to the paths to the labeled images and finally ```path_anno``` corresponds to the annotations from pifpaf.
* run ```python ap_kitti.py```
* run ```python main.py```

## References

For the ```main.py``` file :
```
@INPROCEEDINGS{8594067,
  author={J. {Cartucho} and R. {Ventura} and M. {Veloso}},
  booktitle={2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Robust Object Recognition Through Symbiotic Deep Learning In Mobile Robots}, 
  year={2018},
  pages={2336-2341},
}
``` 
