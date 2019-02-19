import matplotlib
# matplotlib.use("Agg")
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
# from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from base_model import BaseModel
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
from pathlib import Path
import cv2
import os


"""
LabelBinarizer Example:
    labels = ["anger", "contempt", "disgust", "fear", "happiness",
            "neutral","no-face", "none", "sadness", "surprise", "uncertain"]

    label_binner = preprocessing.LabelBinarizer()
    label_binner.fit(labels)
    label_binner.classes_
    label_binner.transform(["anger"])
    label_binner.transform(["surprise", "anger", "sadness"])
"""
# train_data = []
# train_labels = []
# valid_data = []
# valid_labels = []
#
# training_path = "training_set_imgs"
# validation_path = "validation_set_imgs"
#
# epochs = 50
# init_lr = 1e-3
# batch_size = 16
# img_width = 100
# img_height = 100
""" PREPROCESS PROCESS DATA AND IMAGES FOR AN EMOTION"""
def process_data(path, emotion):
    try:
        data = []
        labels = []
        print("Processing ", path, "...")
        emotion_path = "{}/{}".format(path, emotion)

        for img in os.scandir(emotion_path):
            print(img.name)
            # RESIZE IMAGE AND CONVERT IMG TO ARRAY OF PIXELS
            # APPEND IMAGE(PIXEL ARRAY) TO DATA[] AND CORRESPONDING LABEL TO LABELS[]
            image = cv2.imread(img.path)
            image = cv2.resize(image, (img_width, img_height))
            image = img_to_array(image)
            data.append(image)

            labels.append(emotion)
            # print(img.name)

            # Convert data and labels to np arrays
    except:
        print("Error processing training data")
        return False
    return data, labels

# Convert Data/Label Keras Arrays to Numpy Arrays
def convert_to_np_arrays(data, labels):
    """
    type data: List[keras_arrays]
    type: labels: List[keras_arrays]
    rtype: None
    """
    pass

# emotions = ["angerX", "contemptX", "disgustX", "fearX", "happiness",
# "neutral", "no-face", "none", "sadness+25459", "surprise+", "uncertain+"]
"""Emotions too large to train at once must split into batches of ~25k:
    {happiness, neutral, none, no-face,}"""
def process_train_data(path, emotion):
    try:
        data = []
        labels = []

        processed_path = "processed/{}".format(emotion)
        emotion_path = "{}/{}".format(path, emotion)

        # create directory to hold processed imgs if not exists already
        if not os.path.exists(processed_path):
            os.makedirs(processed_path)


        print("Processing ", path, "...")

        # GET LIST OF EMOTION IMAGES
        emotion_imgs = os.listdir(emotion_path)

        for i in range(0, 5000):
            # print(emotion_imgs[i])
            # RESIZE IMAGE AND CONVERT IMG TO ARRAY OF PIXELS
            # APPEND IMAGE(PIXEL ARRAY) TO DATA[] AND CORRESPONDING LABEL TO LABELS[]
            # PATH TO TRAINING_SET_IMGS: training_set_imgs/{emotion}/filename
            img_src = "{}/{}".format(emotion_path, emotion_imgs[i])
            img_dest = "{}/{}".format(processed_path, emotion_imgs[i])

            # print("\n\nimg_src: ", img_src)
            # print("img_dest: ", img_dest)

            # image = cv2.imread(img_src)
            # image = cv2.resize(image, (img_width, img_height))
            # image = img_to_array
            # data.append(image)

            # labels.append(emotion)

            # MOVE IMAGE TO PROCESSED DIRECTORY. (copy for now...)
            if ((not Path(img_dest).exists()) and (os.path.exists(img_src))):
                print("Moving ", emotion_imgs[i])

            # image = cv2.imread(img.path)
            # image = cv2.resize(image, (img_width, img_height))
            # image = img_to_array(image)
            # data.append(image)

            # labels.append(emotion)
            # print(img.name)
    except Exception as e:
        print("Error processing training data")
        print(e)
        return False
    return data, labels

# def create_add_to_pickle(train_pickle_fn, valid_pickle_fn, emotion):
""" ADDS INITIAL DATA/LABELS TO PICKLES
    ONLY RUN IF FIRST TIME CREATING PICKLES"""
def create_add_to_pickles(emotion):
    # ONLY RUN THIS METHOD IF FIRST TIME CREATING PICKLES
        # emotion = "contempt"
    training_path = "training_set_imgs"
    validation_path = "validation_set_imgs"

    valid_data_pickle_fn = "valid_data_pickle"
    valid_label_pickle_fn = "valid_label_pickle"

    train_data_pickle_fn = "train_data_pickle"
    train_label_pickle_fn = "train_label_pickle"

    # VALIDATION SET
    valid_data, valid_labels = process_data(validation_path, emotion)
    print("processed training: ",emotion, "\n", "data: ", len(valid_data), "\n"
    "labels: ", len(valid_labels))

    # valid_data_pickle = open("valid_data_pickle", 'wb')
    valid_data_pickle = open(valid_data_pickle_fn, 'wb')
    pickle.dump(valid_data, valid_data_pickle_fn)
    valid_data_pickle.close()

    valid_label_pickle = open(valid_label_pickle_fn, 'wb')
    pickle.dump(valid_labels, valid_label_pickle_fn)
    valid_label_pickle.close()

    # TRAINING SET
    train_data, train_labels = process_data(training_path, emotion)
    print("processed training: ",emotion, "\n", "data: ", len(train_data), "\n"
    "labels: ", len(train_labels))

    train_data_pickle = open(train_data_pickle_fn, 'wb')
    pickle.dump(train_data, train_data_pickle_fn)
    train_data_pickle.close()

    train_label_pickle = open(train_label_pickle_fn, 'wb')
    pickle.dump(train_labels, train_label_pickle_fn)
    train_label_pickle.close()

    return True

""" LOAD AND READ FROM PICKLE FILES
    DISPLAYS LENGTH OF VALID AND TRAINING PICKLES"""
def load_read_pickles(valid_pickle, valid_label_pickle, train_pickle, train_label_pickle):
    # LOAD/READ FROM VALID PICKLE FILES
        # VALIDATION SET
    # valid_pickle = "valid_data_pickle"
    valid_data_pickle_load = open(valid_pickle, 'rb')
    valid_data = pickle.load(valid_data_pickle_load)
    print("\n# valid data pickle entries: ", len(valid_data))

    # valid_label_pickle = "valid_label_pickle"
    valid_label_pickle_load = open(valid_label_pickle, 'rb')
    valid_labels = pickle.load(valid_label_pickle_load)
    print("# valid label pickle entries: ", len(valid_labels))

    # LOAD/READ FROM TRAIN PICKLE FILES
        # TRAINING SET
    # train_pickle = "train_data_pickle"
    train_data_pickle_load = open(train_pickle, 'rb')
    train_data = pickle.load(train_data_pickle_load)
    print("\n\n# train data pickle entries: ", len(train_data))

    # train_label_pickle = "train_label_pickle"
    train_label_pickle_load = open(train_label_pickle, 'rb')
    train_labels = pickle.load(train_label_pickle_load)
    print("# train label pickle entries: ", len(train_labels))

    return True

""" APPENDS TO EXISTING PICKLE FILES
    RUN THIS METHOD AFTER FIRST CREATION OF PICKLE FILES"""
def append_to_pickle(path, emotion, data_pickle, label_pickle):
    # emotion = "happiness" # TODO: HAPPINESS

    data = []
    labels = []


        # VALIDATION SET
    # valid_data, valid_labels = process_data(validation_path, emotion, valid_data, valid_labels)
    data, labels = process_data(path, emotion)
    print("processed training: ",emotion, "\n", "data: ", len(data), "\n"
            "labels: ", len(labels))

    # OPEN AND LOAD DATA PICKLE FILE TO DATA_PICKLE_LOAD
    # valid_pickle = "valid_data_pickle"
    # valid_data_pickle_load = open(valid_pickle, 'rb')
    data_pickle_load = open(data_pickle, 'rb')
    data_pickle_list = pickle.load(data_pickle_load)
    # APPEND PROCESSED DATA TO LOADED DATA_PICKLE
    data_pickle_list += data
    data_pickle_load.close()

    # SAVE UPDATED DATA TO DATA_PICKLE FILE
    # valid_data_pickle_file = open("valid_data_pickle", 'wb')
    data_pickle_file = open(data_pickle, 'wb')
    pickle.dump(data_pickle_list, data_pickle_file)
    data_pickle_file.close()

    # OPEN AND LOAD LABEL PICKLE FILE TO LABEL_PICKLE_LOAD
    # valid_label_pickle = "valid_label_pickle"
    # valid_label_pickle_load = open(valid_label_pickle, 'rb')
    label_pickle_load = open(label_pickle, 'rb')
    label_pickle_list = pickle.load(label_pickle_load)
    # APPEND PROCESSED LABELS TO LOADED LABEL_PICKLE
    label_pickle_list += labels
    label_pickle_load.close()

    # SAVE UPDATED LABELS TO LABEL_PICKLE_FILE
    # valid_label_pickle_file = open("valid_label_pickle", 'wb')
    label_pickle_file = open(label_pickle, 'wb')
    pickle.dump(label_pickle_list, label_pickle_file)
    label_pickle_file.close()

    return True


if __name__ == "__main__":
    train_data = []
    train_labels = []
    valid_data = []
    valid_labels = []

    # training_path = "training_set_imgs"
    # validation_path = "validation_set_imgs"

    epochs = 50
    init_lr = 1e-3
    batch_size = 16
    img_width = 100
    img_height = 100

    """
        Process Training Images In Batches of 5k
    """
    # process_train_data("training_set_imgs", "happiness")

    # CREATE/ADD TO VALID PICKLE FILE
    # emotion = "contempt"
    """create_add_to_pickles(emotion)"""

    # LOAD/READ FROM VALID PICKLE FILE

    valid_pickle = "valid_data_pickle"
    valid_label_pickle = "valid_label_pickle"

    train_pickle = "train_data_pickle"
    train_label_pickle = "train_label_pickle"
    load_read_pickles(valid_pickle, valid_label_pickle, train_pickle, train_label_pickle)


    # APPEND TO VALID PICKLE
    # emotions = ["anger24882_X", "contempt3750_X", "disgust3803_X", "fear6378_X", "happiness+134416",
    # "neutral+74874", "no-face+82415", "none+33088", "sadness+25459_X", "surprise+14090_X", "uncertain+11645_X"]

    emotion = "uncertain" # TODO: HAPPINESS
    training_path = "training_set_imgs"
    validation_path = "validation_set_imgs"

    valid_pickle = "valid_data_pickle"
    train_pickle = "train_data_pickle"

    valid_label_pickle = "valid_label_pickle"
    train_label_pickle = "train_label_pickle"

    # append_to_pickle(training_path, emotion, train_pickle, train_label_pickle)
    # append_to_pickle(validation_path, emotion, valid_pickle, valid_label_pickle)
