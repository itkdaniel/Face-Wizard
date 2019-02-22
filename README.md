# Face Wizard
A CMPS140 Artificial Intelligence Project
# Getting Started
To get the datasets refer to the [Datasets](#Datasets) section of this document.  

## Affectnet Dataset
### Unzip the dataset
Request access to [Affectnet Dataset](#Datasets) and download the Manually_Annotated and Manually_Annotated_file_lists zip files. 

The zip file contains the manually annotated images and csv files for training and validation sets. 

Uncompress the zip files and put the Manually_Annotated folder, training.csv file, and validation.csv file into the root project directory.

### Setup Affectnet Dataset
1. Build the emotion:List[images] hash pickle files  
`$python organize_affectdb.py -b`

2. Organize validation set images  
`$python organize_affectdb.py -o validation_set`

3. Organize training set images  
`$python organize_affectdb.py -o training_set`

The above commands will setup the training and validation set images into their corresponding folders with 11 emotions:

 > **anger, contempt, disgust, fear, happiness, neutral, no-face, none, sadness, surprise, uncertain**
 
### Quick build
**Instead of pre-processing the data manually**.  
This will avoid having to manually preprocess each emotion and will automatically quickly preprocess a fixed amount of data, build, and train the model.  
`$python train_model.py --fast`

**Otherwise:**

Follow the [instructions](#Pre-process the data) below to manually preprocess, build, and train the model.
 
### Pre-process the data
#### Validation Set
1. `$python --create validation_set --emotion anger`
2. `$python --append validation_set validation_set_data_pickle validation_set_label_pickle --emotion contempt`
3. Repeat the --append command for the remaining emotions

#### Training Set
1. `$python --createbatch training_set --emotion anger`
2. `$python --appendlarge training_set training_set_data_pickle training_set_label_pickle --contempt`
3. Repeat the --appendlarge command for the remaining emotions  
 **Change the `batch_size` variable in preprop_data.py to control how many images get processed for an emotion**

### Load and read data size metrics
1. `python preprop_data.py --load validation_set_data_pickle validation_set_label_pickle training_set_data_pickle training_set_label_pickle`

### Train the model after manual preprocessing (the long way)
**Currently**:  
1. `$python train_model.py --long`

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
