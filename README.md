# Face Wizard
A CMPS140 Artificial Intelligence Project
# Getting Started
To get the datasets refer to the [Datasets](#Datasets) section of this document.  

## Affectnet Dataset
### Unzip the dataset
Request access to [Affectnet Dataset](#Datasets) and download the Manually Annotated zip file. 

The zip file contains the manually annotated images and csv files for training and validation sets in their corresponding directories. 

Uncompress the zip file and put the Manually_Annotated folder, training.csv file, and validation.csv file into the root project directory.

### Setup Affectnet Dataset
1. Build the emotion:List[images] hash pickle files  
`$python organize_affectdb.py -b`

2. Organize validation set images  
`$python organize_affectdb.py -o validation_set`

3. Organize training set images  
`$python organize_affectdb.py -o training_set`

The above commands will setup the training and validation set images into their corresponding folders with 11 emotions:

 > **anger, contempt, disgust, fear, happiness, neutral, no-face, none, sadness, surprise, uncertain**

## Cohn-Kanade Dataset
To get the dataset refer to **get_ck_dataset.readme**.
### Unzip the dataset
Download the Emotion_labels, extended-cohn-kanade-images, and FACS_labels zipped archives from the **CK+** directory.    

Move Emotion_labels, extended-cohn-kanade-images, and FACS_labels zip files into root project directory.

Unzip the 3 compressed files. There should now be 3 folders (Emotion_labels, extended-cohn-kanade-images, and FACS_labels) in the root project directory.

### Compartmentalize the images
To organize the dataset into emotion categories:

`$python organize_dataset.py`

This will organize and sort images from the dataset into separate folders by emotions:

> **neutral, anger, contempt, disgust, fear, happy, surprise**

## Contributors
Daniel Truong  
Ruiwen Liang  
Michael Lau  


## Datasets
* [Affectnet Facial Expression Database](http://mohammadmahoor.com/affectnet/)
* [Cohn-Kanade AU-Coded Expression Database](http://www.pitt.edu/~emotion/ck-spread.htm)
<!--* [The Japanese Female Facial Expression (JAFFE) Database](http://www.kasrl.org/jaffe.html) 
* [Indian Movie Face Database (IMFDB)](http://cvit.iiit.ac.in/projects/IMFDB/) %}
 -->
