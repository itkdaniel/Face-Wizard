# import matplotlib
# matplotlib.use("Agg")
# from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
# from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import preprocessing
# from sklearn.model_selection import train_test_split
# from base_model import BaseModel
# import matplotlib.pyplot as plt
from imutils import paths
from timeit import default_timer as timer
import numpy as np
import argparse
import random
import pickle
from pathlib import Path
import cv2
import sys
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
        img_width = 256
        img_height = 256

        print("Processing ", path, "...")

        emotion_path = "{}/{}".format(path, emotion)

        for img in os.scandir(emotion_path):
            # print(img.name)
            # RESIZE IMAGE AND CONVERT IMG TO ARRAY OF PIXELS
            # APPEND IMAGE(PIXEL ARRAY) TO DATA[] AND CORRESPONDING LABEL TO LABELS[]
            image = cv2.imread(img.path)
            image = cv2.resize(image, (img_width, img_height))
            image = img_to_array(image)
            data.append(image)

            labels.append(emotion)
            # print(img.name)

            # CANNOT APPEND NP ARRAYS TO PICKLE FILES
            # MUST CONVER TO NP ARRAY IN A SEPARATE METHOD AFTER FINISHED PREPROCESING ALL DATA.
            # Convert the FINAL data and labels list to np arrays
        # data = np.array(data, dtype="float") / 255.0
        # labels = np.array(labels)
        #
        # print(data)
        #
        # binarizer = preprocessing.LabelBinarizer()
        # labels = binarizer.fit_transform(labels)

        # print(labels)


        # print("[INFO] data matrix: {:.2f}MB".format(data.nbytes / (1024 * 1000.0)))
    except Exception as e:
        print("\nError processing", path, "data")
        print(e)
        print()
    return data, labels

# emotions = ["angerX", "contemptX", "disgustX", "fearX", "happiness",
# "neutral", "no-face", "none", "sadness+25459", "surprise+", "uncertain+"]
"""Emotions too large to train at once must split into batches of ~25k:
    {happiness, neutral, none, no-face,}"""
def process_large_data(path, emotion):
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


        for i in range(0, 450):
            # print(emotion_imgs[i])
            # RESIZE IMAGE AND CONVERT IMG TO ARRAY OF PIXELS
            # APPEND IMAGE(PIXEL ARRAY) TO DATA[] AND CORRESPONDING LABEL TO LABELS[]
            # PATH TO TRAINING_SET_IMGS: training_set_imgs/{emotion}/filename
            img_src = "{}/{}".format(emotion_path, emotion_imgs[i])
            # img_dest = "{}/{}".format(processed_path, emotion_imgs[i])

            # print("\n\nimg_src: ", img_src)
            # print("img_dest: ", img_dest)

            image = cv2.imread(img_src)
            image = cv2.resize(image, (img_width, img_height))
            image = img_to_array(image)
            data.append(image)

            labels.append(emotion)

            # TODO: MOVE IMAGE TO PROCESSED DIRECTORY. (copy for now...)
            # if ((not Path(img_dest).exists()) and (os.path.exists(img_src))):
                # print("Moving ", emotion_imgs[i])
    except Exception as e:
        print("Error processing %s data" % path)
        print(e)
        print()
    return data, labels

# emotions = ["angerX", "contemptX", "disgustX", "fearX", "happiness",
# "neutral", "no-face", "none", "sadness+25459", "surprise+", "uncertain+"]
"""Emotions too large to train at once must split into batches of ~25k:
    {happiness, neutral, none, no-face}"""
def process_fixed_data(path, batch_size):
    try:
        data = []
        labels = []

        emotions = ["anger", "contempt", "disgust", "fear", "happiness",
                  "neutral", "no-face", "none", "sadness", "surprise", "uncertain"]

        random_seed_fn = "process_seed.pickle"
        seed_val = -1

        # Read previous seed if exists
        if (os.path.exists(random_seed_fn)):
            random_seed_pickle = open(random_seed_fn, 'rb')
            prev_seed_val = pickle.load(random_seed_pickle)
            prev_seed_val += 1
            random_seed_pickle.close()
        else:
            seed_val += 1


        # Randomize and process all images
        # random.seed()
        random.shuffle(emotions)

        print("\nProcessing data...")

        for emotion in emotions:

            # processed_path = "processed/{}".format(emotion)
            emotion_path = "{}/{}".format(path, emotion)

            # create directory to hold processed imgs if not exists already
            # if not os.path.exists(processed_path):
                # os.makedirs(processed_path)

            print("\nProcessing %s: %s" % path, emotion)

            # GET LIST OF EMOTION IMAGES
            emotion_imgs = os.listdir(emotion_path)

            # Shuffle data
            random.seed(seed_val)
            random.shuffle(emotion_imgs)


            for i in range(0, batch_size):
                # print(emotion_imgs[i])
                # RESIZE IMAGE AND CONVERT IMG TO ARRAY OF PIXELS
                # APPEND IMAGE(PIXEL ARRAY) TO DATA[] AND CORRESPONDING LABEL TO LABELS[]
                # PATH TO TRAINING_SET_IMGS: training_set_imgs/{emotion}/filename
                img_src = "{}/{}".format(emotion_path, emotion_imgs[i])
                # img_dest = "{}/{}".format(processed_path, emotion_imgs[i])

                # print("\n\nimg_src: ", img_src)
                # print("img_dest: ", img_dest)

                image = cv2.imread(img_src)
                image = cv2.resize(image, (img_width, img_height))
                image = img_to_array(image)
                data.append(image)

                labels.append(emotion)

                # TODO: MOVE IMAGE TO PROCESSED DIRECTORY. (copy for now...)
                # if ((not Path(img_dest).exists()) and (os.path.exists(img_src))):
                    # print("Moving ", emotion_imgs[i])

            # Save the current random_seed val
            random_seed_pickle = open(random_seed_fn, 'wb')
            pickle.dump(seed_val, random_seed_pickle)
            random_seed_pickle.close()

    except Exception as e:
        print("Error processing %s data" % path)
        print(e)
        print()
    return data, labels

# Convert Data/Label Keras Arrays to Numpy Arrays
# def convert_to_np_arrays(data, labels):
def convert_to_np_arrays(image_set):
    """
    type image_set: String : image set to convert (training_set, validation_set)
    type data: List[keras_arrays]
    type: labels: List[keras_arrays]
    rtype: None
    """

    # Define pickle file names using the specified image set (training_set, validation_set)
    data_pickle_fn = "{}_data_pickle".format(image_set)
    label_pickle_fn = "{}_label_pickle".format(image_set)

    # Define np array pickle file names
    data_np_pickle_fn = "{}_data_np_pickle".format(image_set)
    label_np_pickle_fn = "{}_label_np_pickle".format(image_set)

    # Load, convert, and save the data pickle
    print("\nConverting processed data(images) to np array...")
    data_pickle_file = open(data_pickle_fn, 'rb')
    data_pickle = pickle.load(data_pickle_file)
    data_pickle_file.close()

    data = np.array(data_pickle, dtype="float") / 255.0

    # Save to pickle file
    print("\n Saving to pickle file...")
    # data_pickle_file = open(data_np_pickle_fn, 'wb')
    # pickle.dump(data, data_pickle_file)
    # data_pickle_file.close()
    # del data


    # Load, convert, and save the label pickle
    print("\nConverting processed labels to np array...")
    label_pickle_file = open(label_pickle_fn, 'rb')
    label_pickle = pickle.load(label_pickle_file)
    label_pickle_file.close()

    labels = np.array(label_pickle)
    binarizer = preprocessing.LabelBinarizer()
    labels = binarizer.fit_transform(labels)

    # Save to pickle file
    print("Saving to pickle file...")
    # label_pickle_file = open(label_np_pickle_fn, 'wb')
    # pickle.dump(labels, label_pickle_file)
    # label_pickle_file.close()
    # del labels

    return data, labels

def create_fixed_shuffle(path, batch_size):
    """
    type: path string: (training_set, validation_set)
    type batch_size int: num of images for each emotion set
    """
    data_pickle_fn = "{}_data_pickle_{}".format(path, batch_size)
    label_pickle_fn = "{}_label_pickle_{}".format(path, batch_size)

    data, labels = process_fixed_data()


    pass



""" ADDS INITIAL DATA/LABELS TO PICKLES
    ONLY RUN IF FIRST TIME CREATING PICKLES"""
def create_add_to_pickles(path, emotion):
    """
    type path: string: "training_set" | "validation_set"
    type emotion: string: "emotion"
    """
    # ONLY RUN THIS METHOD IF FIRST TIME CREATING PICKLES
        # emotion = "contempt"
    # training_path = "training_set_imgs"
    # validation_path = "validation_set_imgs"

    data_pickle_fn = "{}_data_pickle".format(path)
    label_pickle_fn = "{}_label_pickle".format(path)

    data, labels = process_data(path, emotion)
    print("processed training: ",emotion, "\n", "data: ", len(data), "\n"
    "labels: ", len(labels))

    print("\nSaving to pickle files...")

    # valid_data_pickle = open("valid_data_pickle", 'wb')
    data_pickle = open(data_pickle_fn, 'wb')
    pickle.dump(data, data_pickle)
    data_pickle.close()

    label_pickle = open(label_pickle_fn, 'wb')
    pickle.dump(labels, label_pickle)
    label_pickle.close()

    return None

def create_add_to_batch_pickles(path, emotion):
    """
    type path: string: "training_set" | "validation_set"
    type emotion: string: "emotion"
    """
    # ONLY RUN THIS METHOD IF FIRST TIME CREATING PICKLES
        # emotion = "contempt"
    # training_path = "training_set_imgs"
    # validation_path = "validation_set_imgs"

    data_pickle_fn = "{}_data_pickle".format(path)
    label_pickle_fn = "{}_label_pickle".format(path)

    data, labels = process_large_data(path, emotion)
    print("processed training: ",emotion, "\n", "data: ", len(data), "\n"
    "labels: ", len(labels))

    print("\nSaving to pickle files...")

    # valid_data_pickle = open("valid_data_pickle", 'wb')
    data_pickle = open(data_pickle_fn, 'wb')
    pickle.dump(data, data_pickle)
    data_pickle.close()

    label_pickle = open(label_pickle_fn, 'wb')
    pickle.dump(labels, label_pickle)
    label_pickle.close()

    return None

""" LOAD AND READ FROM PICKLE FILES
    DISPLAYS LENGTH OF VALID AND TRAINING PICKLES"""
def load_read_pickles(valid_pickle, valid_label_pickle, train_pickle, train_label_pickle):
    # LOAD/READ FROM VALID PICKLE FILES
        # VALIDATION SET
    # valid_pickle = "valid_data_pickle"
    data_pickle_load = open(valid_pickle, 'rb')
    valid = pickle.load(data_pickle_load)
    data_pickle_load.close()
    print("\n# valid data pickle entries: ", len(valid))

    # valid_label_pickle = "valid_label_pickle"
    label_pickle_load = open(valid_label_pickle, 'rb')
    valid = pickle.load(label_pickle_load)
    label_pickle_load.close()
    print("# valid label pickle entries: ", len(valid))
    del valid

    # LOAD/READ FROM TRAIN PICKLE FILES
        # TRAINING SET
    # train_pickle = "train_data_pickle"
    data_pickle_load = open(train_pickle, 'rb')
    train = pickle.load(data_pickle_load)
    data_pickle_load.close()
    print("\n\n# train data pickle entries: ", len(train))

    # train_label_pickle = "train_label_pickle"
    label_pickle_load = open(train_label_pickle, 'rb')
    train = pickle.load(label_pickle_load)
    label_pickle_load.close()
    print("# train label pickle entries: ", len(train))
    del train

    return True

""" APPENDS TO EXISTING PICKLE FILES
    RUN THIS METHOD AFTER FIRST CREATION OF PICKLE FILES"""
def append_to_pickle(path, data_pickle, label_pickle, emotion):
    # emotion = "happiness" # TODO: HAPPINESS

    data = []
    labels = []

        # VALIDATION SET
    # PREPROCESS THE DATA
    # valid_data, valid_labels = process_data(validation_path, emotion, valid_data, valid_labels)
    data, labels = process_data(path, emotion)
    print("processed training: ",emotion, "\n", "data: ", len(data), "\n"
            "labels: ", len(labels))

    print("\nSaving to pickle files...")

    # OPEN AND LOAD DATA PICKLE FILE TO DATA_PICKLE_LOAD
    # valid_pickle = "valid_data_pickle"
    # valid_data_pickle_load = open(valid_pickle, 'rb')
    data_pickle_load = open(data_pickle, 'rb')
    data_pickle_list = pickle.load(data_pickle_load)
    # APPEND PROCESSED DATA TO LOADED DATA_PICKLE
    data_pickle_list += data
    data_pickle_load.close()

    # SAVE UPDATED DATA TO DATA_PICKLE FILE
    data_pickle_load = open(data_pickle, 'wb')
    pickle.dump(data_pickle_list, data_pickle_load)
    data_pickle_load.close()
    del data_pickle_list

    # OPEN AND LOAD LABEL PICKLE FILE TO LABEL_PICKLE_LOAD
    label_pickle_load = open(label_pickle, 'rb')
    label_pickle_list = pickle.load(label_pickle_load)
    # APPEND PROCESSED LABELS TO LOADED LABEL_PICKLE
    label_pickle_list += labels
    label_pickle_load.close()

    # SAVE UPDATED LABELS TO LABEL_PICKLE_FILE
    label_pickle_load = open(label_pickle, 'wb')
    pickle.dump(label_pickle_list, label_pickle_load)
    label_pickle_load.close()
    del label_pickle_list


    return True

def append_large_to_pickle(path, data_pickle, label_pickle, emotion):
    data = []
    labels = []

        # VALIDATION SET
    # PREPROCESS THE DATA
    # valid_data, valid_labels = process_data(validation_path, emotion, valid_data, valid_labels)
    data, labels = process_large_data(path, emotion)
    print("processed training: ",emotion, "\n", "data: ", len(data), "\n"
            "labels: ", len(labels))

    print("\nSaving to pickle files...")

    # OPEN AND LOAD DATA PICKLE FILE TO DATA_PICKLE_LOAD
    data_pickle_load = open(data_pickle, 'rb')
    data_pickle_list = pickle.load(data_pickle_load)
    # APPEND PROCESSED DATA TO LOADED DATA_PICKLE
    data_pickle_list += data
    data_pickle_load.close()

    # SAVE UPDATED DATA TO DATA_PICKLE FILE
    data_pickle_load = open(data_pickle, 'wb')
    pickle.dump(data_pickle_list, data_pickle_load)
    data_pickle_load.close()
    del data_pickle_list

    # OPEN AND LOAD LABEL PICKLE FILE TO LABEL_PICKLE_LOAD
    label_pickle_load = open(label_pickle, 'rb')
    label_pickle_list = pickle.load(label_pickle_load)
    # APPEND PROCESSED LABELS TO LOADED LABEL_PICKLE
    label_pickle_list += labels
    label_pickle_load.close()

    # SAVE UPDATED LABELS TO LABEL_PICKLE_FILE
    label_pickle_load = open(label_pickle, 'wb')
    pickle.dump(label_pickle_list, label_pickle_load)
    label_pickle_load.close()
    del label_pickle_list


    return True

"""
# Emotions used to prepropcess data
# emotions = ["anger+24882_", "contempt+3750_X", "disgust+3803_X", "fear+6378_X", "happiness+134416_X",
#           "neutral+74874_", "no-face+82415_", "none+33088_", "sadness+25459_X", "surprise+14090_", "uncertain+11645_"]
#
# 3750 : contempt
# 3803 : disgust
# 5,000: happiness, sadness, anger, neutral, no-face, surprise
# TODO : none, uncertain
"""
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
    img_width = 256
    img_height = 256

    # emotions = {"anger":True, "contempt":True, "disgust":True, "fear":True, "happiness":True,
    # "neutral":True, "no-face":True, "none":True, "sadness":True, "surprise":True, "uncertain":True}

    # emotions = ["angerX", "contemptX", "disgustX", "fearX", "happiness",
    # "neutral", "no-face", "none", "sadness+25459", "surprise+", "uncertain+"]


    # * add command line arguments:
    #     - create_add_to_pickle
    #     - load_read_pickles
    #     - append_to_pickle
    # Finish implementing argparse for the above methods
    parser = argparse.ArgumentParser()

    parser.add_argument("--create", help="directory for set of images to process: {training_set, validation_set}")
    parser.add_argument("--createbatch", help="directory for set of images to process: {training_set, validation_set}")

    parser.add_argument("--append", help="data to append: {training, validation}", nargs=3)
    parser.add_argument("--appendlarge", help="large data to append", nargs=3)

    parser.add_argument("--convert", help="data to convert to np arrays: {training_set, validation_set}")

    # -load: valid_data_pickle, valid_label_pickle, train_data_pickle, train_label_pickle
    # parser.add_argument("-load", help="load and read from preprocessed pickle files", nargs=4,
    #                     choices=["valid_data_pickle", "valid_label_pickle",
    #                             "train_data_pickle", "train_label_pickle"])
    parser.add_argument("--load", help="load and read from preprocessed pickle files", nargs=4)
    # parser.add_argument("-data", help="type of data to load: {data, labels}")

    parser.add_argument("--emotion", choices=["anger", "contempt", "disgust", "fear", "happiness",
                                            "neutral", "no-face", "none", "sadness", "surprise",
                                            "uncertain"])


    args = parser.parse_args()

    # Alternate method to require command line arguments
    if (len(sys.argv) <= 1):
        parser.print_usage()
        sys.exit()

    # emotions = ["anger+24882_", "contempt+3750_X", "disgust+3803_X", "fear+6378_X", "happiness+134416_",
    #           "neutral+74874_", "no-face+82415_", "none+33088_", "sadness+25459_", "surprise+14090_", "uncertain+11645_"]

    # 10,000 : anger, happiness, neutral, no-face, none, sadness, surprise, uncertain

    # emotions = {"anger":True, "contempt":False, "disgust":False, "fear":False, "happiness":False,
    #             "neutral":False, "no-face":False, "none":False, "sadness":False, "surprise":False, "uncertain":False}

    start = timer()

    """
    COMMAND LINE ARGUMENTS FOR:
        -creating initial pickle
            files of preprocessed data and labels
        -loading and reading from existing pickle files
        -appending to exisiting pickle files"""

    # COMMAND LINE ARGUMENTS FOR CREATING INITIAL PICKLE FILES
    # python preprop_data.py --create training_set --emotion contempt
    # python preprop_data.py --create validation_set --emotion contempt
    # if user specifies a data set (training_set, validation_set)
    # and an emotion to process
    if (args.create):
        print("type create: ", type(args.create))
        if (args.emotion):
            # if (args.emotion in emotions.keys()):
            print("emotion: ", args.emotion)
            # if user specifies training_set images
            if ("train" in args.create):
                create_add_to_pickles(args.create, args.emotion)
                # pass
            # if user specifies validation_set images
            elif ("valid" in args.create):
                create_add_to_pickles(args.create, args.emotion)
                # pass
            else:
                print("Error: Must specify training or validation set")
                parser.print_usage()
                sys.exit()
            # else:
            #     emotion_error_msg = "Error: emotion {} does not exist in dataset".format(args.emotion)
            #     print(emotion_error_msg)
            #     parser.print_usage()
            #     sys.exit()
        else:
            print("Error: Missing an argument")
            parser.print_usage()
            sys.exit()

    # Initial creation of large pickle file
    if (args.createbatch):
        print("type create: ", type(args.createbatch))
        if (args.emotion):
            # if (args.emotion in emotions.keys()):
            print("emotion: ", args.emotion)
            # if user specifies training_set images
            if ("train" in args.createbatch):
                create_add_to_batch_pickles(args.createbatch, args.emotion)
                # pass
            # if user specifies validation_set images
            elif ("valid" in args.createbatch):
                create_add_to_batch_pickles(args.createbatch, args.emotion)
                # pass
            else:
                print("Error: Must specify training or validation set")
                parser.print_usage()
                sys.exit()
            # else:
            #     emotion_error_msg = "Error: emotion {} does not exist in dataset".format(args.emotion)
            #     print(emotion_error_msg)
            #     parser.print_usage()
            #     sys.exit()
        else:
            print("Error: Missing an argument")
            parser.print_usage()
            sys.exit()


    # COMMAND LINE ARGUMENTS TO LOAD AND READ FROM PICKLE FILES
    # python preprop_data.py --load validation_set_data_pickle validation_set_label_pickle training_set_data_pickle training_set_label_pickle
    # arg0 = valid_data_pickle
    # arg1 = valid_label_pickle
    # arg2 = train_data_pickle
    # arg3 = train_label_pickle
    elif (args.load):
        # Check if user correctly specified all (valid, train)-(data, label) pickle file names
        if (("valid" in args.load[0] and "data" in args.load[0] and "pickle" in args.load[0])
            and ("valid" in args.load[1] and "label" in args.load[1] and "pickle" in args.load[1])
            and ("train" in args.load[2] and "data" in args.load[2] and "pickle" in args.load[2])
            and ("train" in args.load[3] and "label" in args.load[3] and "pickle" in args.load[3])):
            # Check if all files user specified exists
            all_exists = True
            if((not os.path.exists(args.load[0]))
                or (not os.path.exists(args.load[1]))
                or (not os.path.exists(args.load[2]))
                or (not os.path.exists(args.load[3]))):
                all_exists = False
            # Display error and exit if not all files exist
            if (not all_exists):
                print("Error: At least of the files in given arguments does not exist")
                parser.print_usage()
                sys.exit()
            # If all files correct and exist -> execute load_read_pickles()
            else:
                print("in args load")
                print("GUUCI MENG ALL VARIABLES GUUUCCIIIII")
                print("loading...")
                load_read_pickles(args.load[0], args.load[1], args.load[2], args.load[3])
        # User did not specify correct pickle files
        else:
            print("Error: Must specify correct (valid, train)-(data, label) pickle files")
            parser.print_usage()
            sys.exit()

    # COMMAND LINE ARGUMENTS TO APPEND TO EXISTING PICKLE FILE
    # python preprop_data.py -append training_set train_data_pickle train_label_pickle -emotion anger
    # python preprop_data.py -append validation_set valid_data_pickle valid_label_pickle  -emotion anger
    elif (args.append):
        # Check if valid file name
        if ("training" in args.append[0] or "validation" in args.append[0]):
            # Check if path to training/validation set images exist
            if (os.path.exists(args.append[0])):
                # Check if pickle files exist
                if (os.path.exists(args.append[1]) and os.path.exists(args.append[2])):
                    # Check if valid emotion specified
                    if (args.emotion):
                        print("in args append")
                        append_to_pickle(args.append[0], args.append[1], args.append[2], args.emotion)
                        # append_to_pickle(training_path, train_data_pickle, train_label_pickle, emotion)
                    else:
                        print("Error: Must specify a valid emotion")
                        parser.print_usage()
                        sys.exit()
                else:
                    print("Error: At least one of the pickle files does not exist")
                    parser.print_usage()
                    sys.exit()
            else:
                print("Error: Path to images does not exist")
                parser.print_usage()
                sys.exit()
        else:
            print("Error: Must specify a correct (training, validation) set path")


    # COMMAND LINE ARGUMENTS TO APPEND LARGE DATA TO EXISTING PICKLE FILE
    # python preprop_data.py --appendlarge training_set train_data_pickle train_label_pickle --emotion anger
    # python preprop_data.py --appendlarge validation_set valid_data_pickle valid_label_pickle  --emotion anger
    elif (args.appendlarge):
        # Check if valid file name
        if ("training" in args.appendlarge[0] or "validation" in args.appendlarge[0]):
            # Check if path to training/validation set images exist
            if (os.path.exists(args.appendlarge[0])):
                # Check if pickle files exist
                if (os.path.exists(args.appendlarge[1]) and os.path.exists(args.appendlarge[2])):
                    # Check if valid emotion specified
                    if (args.emotion):
                        print("in args appendlarge")
                        append_large_to_pickle(args.appendlarge[0], args.appendlarge[1], args.appendlarge[2], args.emotion)
                        # append_to_pickle(training_path, train_data_pickle, train_label_pickle, emotion)
                    else:
                        print("Error: Must specify a valid emotion")
                        parser.print_usage()
                        sys.exit()
                else:
                    print("Error: At least one of the pickle files does not exist")
                    parser.print_usage()
                    sys.exit()
            else:
                print("Error: Path to images does not exist")
                parser.print_usage()
                sys.exit()
        else:
            print("Error: Must specify a correct (training, validation) set path")

    elif (args.convert):
        if("training" in args.convert or "validation" in args.convert):
            convert_to_np_arrays(args.convert)
        else:
            print("Error: Please specify valid data or labels pickles to convert")
            parser.print_usage()
            sys.exit()




    finish = timer() - start
    print("\nExecution took %.2f seconds" % finish)

    """
        Process Training Images In Batches of 5k
    """
    # process_train_data("training_set_imgs", "happiness")

    # CREATE/ADD TO VALID PICKLE FILE
    # emotion = "contempt"
    # if args.create and args.emotion:
    #     print(args.create)
    #     print(args.emotion)

    # create_add_to_pickles("training_set", emotion)

    # LOAD/READ FROM VALID PICKLE FILES
"""
    valid_pickle = "valid_data_pickle"
    valid_label_pickle = "valid_label_pickle"

    train_pickle = "train_data_pickle"
    train_label_pickle = "train_label_pickle"
    load_read_pickles(valid_pickle, valid_label_pickle, train_pickle, train_label_pickle)
"""

    # APPEND TO VALID PICKLE
    # emotions = ["anger24882_X", "contempt3750_X", "disgust3803_X", "fear6378_X", "happiness+134416",
    # "neutral+74874", "no-face+82415", "none+33088", "sadness+25459_X", "surprise+14090_X", "uncertain+11645_X"]

"""
    emotion = "uncertain" # TODO: HAPPINESS
    training_path = "training_set_imgs"
    validation_path = "validation_set_imgs"

    valid_pickle = "valid_data_pickle"
    train_pickle = "train_data_pickle"

    valid_label_pickle = "valid_label_pickle"
    train_label_pickle = "train_label_pickle"
"""
    # append_to_pickle(training_path, emotion, train_pickle, train_label_pickle)
    # append_to_pickle(validation_path, emotion, valid_pickle, valid_label_pickle)
