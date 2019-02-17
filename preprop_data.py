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
        # for dir in os.scandir(training_path):
        #     print(dir.name)
        #     for img in os.scandir(dir.path):
        #         # print(img.name)
        #         image = cv2.imread(img.path)
        #         image = cv2.resize(image, (img_width, img_height))
        #         image = img_to_array(image)
        #         data.append(image)
        #
        #         label = dir.name
        #         labels.append(label)
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
    except:
        print("Error processing training data")
        return False
    return data, labels

# def create_add_to_pickle(train_pickle_fn, valid_pickle_fn, emotion):
""" ADDS INITIAL DATA/LABELS TO PICKLES
    ONLY RUN IF FIRST TIME CREATING PICKLES"""
def create_add_to_pickles(emotion):
    """WARNING: Some emotions may contain too many images to process all at once (ex: happiness > 100k images)
                **NEED MORE RAM** """
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
    emotion = "happiness" # TODO: HAPPINESS

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
    data_pickle = pickle.load(data_pickle_load)
    # APPEND PROCESSED DATA TO LOADED DATA_PICKLE
    data_pickle += data
    data_pickle_load.close()

    # SAVE UPDATED DATA TO DATA_PICKLE FILE
    # valid_data_pickle_file = open("valid_data_pickle", 'wb')
    data_pickle_file = open(data_pickle, 'wb')
    pickle.dump(data_pickle, data_pickle_file)
    data_pickle_file.close()

    # OPEN AND LOAD LABEL PICKLE FILE TO LABEL_PICKLE_LOAD
    # valid_label_pickle = "valid_label_pickle"
    # valid_label_pickle_load = open(valid_label_pickle, 'rb')
    label_pickle_load = open(label_pickle, 'rb')
    label_pickle = pickle.load(label_pickle_load)
    # APPEND PROCESSED LABELS TO LOADED LABEL_PICKLE
    label_pickle += labels
    label_pickle_load.close()

    # SAVE UPDATED LABELS TO LABEL_PICKLE_FILE
    # valid_label_pickle_file = open("valid_label_pickle", 'wb')
    label_pickle_file = open(label_pickle, 'wb')
    pickle.dump(label_pickle, label_pickle_file)
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

    # CREATE/ADD TO VALID PICKLE FILE
    # UNCOMMENT BELOW TO CREATE/ADD TO PICKLES
    """
    emotion = "contempt"
    create_add_to_pickles(emotion)
    """

    # LOAD/READ FROM VALID PICKLE FILE
    valid_pickle = "valid_data_pickle"
    valid_label_pickle = "valid_label_pickle"

    train_pickle = "train_data_pickle"
    train_label_pickle = "train_label_pickle"
    load_read_pickles(valid_pickle, valid_label_pickle, train_pickle, train_label_pickle)

# emotions = ["anger", "contempt", "disgust", "fear", "happiness",
# "neutral", "no-face", "none", "sadness", "surprise", "uncertain"]

    # APPEND TO VALID PICKLE
    # UNCOMMENT BELOW TO APPEND TO PICKLES
    """
    # emotion = "happiness" # TODO: HAPPINESS
    training_path = "training_set_imgs"
    validation_path = "validation_set_imgs"

    valid_pickle = "valid_data_pickle"
    train_pickle = "train_data_pickle"

    valid_label_pickle = "valid_label_pickle"
    train_label_pickle = "train_label_pickle"

    append_to_pickle(training_path, emotion, valid_pickle, valid_label_pickle)
    append_to_pickle(validation_path, emotion, train__pickle, train_label_pickle)
    """
