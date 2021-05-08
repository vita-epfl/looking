# Ap detection

## On JAAD test set

```
chmod +x ap_jaad.sh
```

Then modify the paths in the shell script

| Parameter                 |Description   |	
| :------------------------ |:-------------|
| PATH_JAAD_REPO  | Path to the offcial JAAD repo |
| PATH_JAAD_TEXT_FILE | Path to the text files created after creating our custom JAAD (default : ```../create_data/splits_jaad```) |

Run :

```
./ap_jaad.sh
```

## On Kitti dataset

```
chmod +x ap_kitti.sh
```
Then modify the paths in the shell script

| Parameter                 |Description   |	
| :------------------------ |:-------------|
| PATH_OUT  | Path where to donwload the offcial gt and our annotations on the test set |
| PATH_KITTI | Path to the downloaded Kitti test set |

Run :

```
./ap_kitti.sh
```

