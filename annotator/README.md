# YouLook - Annotation tool

<p align="center">
  <img align="center" src="https://github.com/younesbelkada/YouLook/blob/main/logo.png" width=30% height=50%>
</p>

This tool has been used to annotate quickly image for looking / not looking benchmark. The User Interface is user-friendly and easy to understand and use. 

## Requirements

```
openpifpaf
pyqt5
opencv-python==4.2.0.32
matplotlib
```

## Install 

To install the tool, just run the following :

```
pip3 install -r requirements.txt
```

## Load the dataset

If you want to contribute to the benchmark to enhance the training set, you can download challenging pictures here : https://drive.google.com/drive/folders/1rJXOAbsVpwafi6Eo7syVosr6QVXEP0j2?usp=sharing

## Usage

Launch the program (run the following on a terminal)
```
git clone https://github.com/younesbelkada/YouLook 
cd YouLook
pip3 install --upgrade pip && pip3 install -r requirements.txt
python3 predict.py
````
Choose the folder to label, press ok and go grab a coffee. The program is running pifpaf on the selected folder (~10 minutes). Once this is done, type in the terminal
```
python3 main.py
```
In the User-Interface: 

* Go to File -> Open -> choose your folder
* You will see the guesses of the current model: green box => looking, Red box => not-looking

How to use the annotator

* Left-click to change the binary label looking / not-looking
* Right-click if you are not sure (yellow box)
* Right-click again to remove the annotation

Keyboard shortcuts

* a, d → move back and forth between images
* s → show the image without annotations
* w → predict again with pifpaf

Once you are done with the labeling, the program will inform you. Please send back the annotation folder + the result folder (anno_<folder_name> + out_<folder_name>) as a zip file.

Many thanks for the help!

For any enquiry, please contact: younes.belkada@epfl.ch
 
## Remarks
  * If you saved a prediction where you removed some instances, when accessing back to the image the software will re-predict again everything.
  * If you removed an instance by accident, you can regenerate the prediction using the ```Predict``` button.
  * The annotations will be saved everytime you go to the next image.
  

