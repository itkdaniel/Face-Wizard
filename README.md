# Face Wizard
A CMPS140 Artificial Intelligence Project
# Getting Started
To get the dataset refer to **get_dataset_README**.  

## Unzip the dataset
Download the Emotion_labels, extended-cohn-kanade-images, and FACS_labels zipped archives from the **CK+** directory.    

Move `Emotion_labels` `extended-cohn-kanade-images` and `FACS_labels` into project root directory (i.e. same directory as **organize_dataset.py**)  

Unzip the 3 compressed datasets. There should now be 3 folders (Emotion_labels, extended-cohn-kanade-images, and FACS_labels) in the **same directory** as organize_dataset.py.   

## Compartmentalize the images
To organize the dataset into emotion categories:

`$ python organize_dataset.py`

This will organize and sort images from the dataset into separate folders by emotions:
* neutral
* anger
* contempt
* disgust
* fear
* happy
* surprise

## Contributors
Ruiwen Liang  
Michael Lau  
Daniel Truong  
