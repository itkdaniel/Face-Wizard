from pathlib import Path
import os, os.path
import datetime
import argparse
import shutil
import pickle
import time
import sys
import csv

"""
Emotion Codes:
0: Neutral, 1: Happiness, 2: Sadness, 3: Surprise, 4: Fear,
5: Disgust, 6: Anger, 7: Contempt, 8: None, 9: Uncertain, 10: No-Face
"""

class DataOrganizer(object):
    # organize data into training set
    def get_training_set(self, path, emotion, emotion_hash):
        """
        :type path: str
        :tpye emotion_hash: Dict{emotion:List[]}
        :rtype: bool
        """
        num_imgs_copied = 0
        # path: "training_set_imgs"
        path_to_training_emotion = "{}/{}".format(path, emotion)

        # create directory to hold emotion imgs if not exists already
        if not os.path.exists(path_to_training_emotion):
            os.makedirs(path_to_training_emotion)

        # for each image in the emotion's image list
        for img in emotion_hash[emotion]:
            img_filename = "{}.png".format(img[0].split('/')[1])
            path_img_src = "{}".format(img[1])
            path_img_dest = "{}/{}".format(path_to_training_emotion, img_filename)
            # if the image does not exist in the destination
            if ((not Path(path_img_dest).exists()) and (os.path.exists(path_img_src))): # before fix: 102,032 happiness files
                # cp the image to the corresponding directory
                # in the training_set_imgs folder
                shutil.copy(path_img_src, path_img_dest)
                num_imgs_copied += 1
                # print("num imgs copied: ", num_imgs_copied)

        # pass
        return num_imgs_copied

    # organize images into emotion folders
    def get_emotion_folders(self, path, csv_training_file):
        # path: "Manually_Annotated\Manually_Annotated_Images"
        """
        :type path: str
        :rtype emotion_hash: Dictionary{}
        """
        emotion_hash = {}
        # read in the csv training file
        with open(csv_training_file) as training_csv:
            csv_read = csv.reader(training_csv, delimiter=',')
            # skip first line
            next(csv_read)
            # for each row in the csv file
            for row in csv_read:
                # get the sub directory and emotion label
                img_subdir = row[0]
                img_path = "{}/{}".format(path, img_subdir)
                img_emo_label = float(row[6])

                # map (img_path, img_emo_label) to corresponding emotion list
                if (img_emo_label == 0):
                    # neutral
                    if ("neutral" not in emotion_hash.keys()):
                        emotion_hash["neutral"] = [(img_subdir, img_path, img_emo_label)]
                    else:
                        emotion_hash["neutral"].append((img_subdir, img_path, img_emo_label))
                elif (img_emo_label == 1):
                    # happiness
                    if ("happiness" not in emotion_hash.keys()):
                        emotion_hash["happiness"] = [(img_subdir, img_path, img_emo_label)]
                    else:
                        emotion_hash["happiness"].append((img_subdir, img_path, img_emo_label))

                elif (img_emo_label == 2):
                    # sadness
                    if ("sadness" not in emotion_hash.keys()):
                        emotion_hash["sadness"] = [(img_subdir, img_path, img_emo_label)]
                    else:
                        emotion_hash["sadness"].append((img_subdir, img_path, img_emo_label))
                elif (img_emo_label == 3):
                    # surprise
                    if ("surprise" not in emotion_hash.keys()):
                        emotion_hash["surprise"] = [(img_subdir, img_path, img_emo_label)]
                    else:
                        emotion_hash["surprise"].append((img_subdir, img_path, img_emo_label))
                elif (img_emo_label == 4):
                    # fear
                    if ("fear" not in emotion_hash.keys()):
                        emotion_hash["fear"] = [(img_subdir, img_path, img_emo_label)]
                    else:
                        emotion_hash["fear"].append((img_subdir, img_path, img_emo_label))
                elif (img_emo_label == 5):
                    # disgust
                    if ("disgust" not in emotion_hash.keys()):
                        emotion_hash["disgust"] = [(img_subdir, img_path, img_emo_label)]
                    else:
                        emotion_hash["disgust"].append((img_subdir, img_path, img_emo_label))
                elif (img_emo_label == 6):
                    # anger
                    if ("anger" not in emotion_hash.keys()):
                        emotion_hash["anger"] = [(img_subdir, img_path, img_emo_label)]
                    else:
                        emotion_hash["anger"].append((img_subdir, img_path, img_emo_label))
                elif (img_emo_label == 7):
                    # contempt
                    if ("contempt" not in emotion_hash.keys()):
                        emotion_hash["contempt"] = [(img_subdir, img_path, img_emo_label)]
                    else:
                        emotion_hash["contempt"].append((img_subdir, img_path, img_emo_label))
                elif (img_emo_label == 8):
                    # none
                    if ("none" not in emotion_hash.keys()):
                        emotion_hash["none"] = [(img_subdir, img_path, img_emo_label)]
                    else:
                        emotion_hash["none"].append((img_subdir, img_path, img_emo_label))
                elif (img_emo_label == 9):
                    # uncertain
                    if ("uncertain" not in emotion_hash.keys()):
                        emotion_hash["uncertain"] = [(img_subdir, img_path, img_emo_label)]
                    else:
                        emotion_hash["uncertain"].append((img_subdir, img_path, img_emo_label))
                elif (img_emo_label == 10):
                    # no-face
                    if ("no-face" not in emotion_hash.keys()):
                        emotion_hash["no-face"] = [(img_subdir, img_path, img_emo_label)]
                    else:
                        emotion_hash["no-face"].append((img_subdir, img_path, img_emo_label))

        return emotion_hash

if __name__ == "__main__":
    csv_training_file = "training.csv"
    csv_validation_file = "validation.csv"
    training_path = "training_set_imgs"
    validation_path = "validation_set_imgs"
    images_path = "Manually_Annotated/manually_Annotated_Images"
    emotions = ["neutral", "happiness", "sadness", "surprise","fear",
                "disgust", "anger", "contempt", "none", "uncertain", "no-face"]
    # emotions = ["neutral", "sadness", "surprise","fear",
                # "disgust", "anger", "contempt", "none", "uncertain", "no-face"]
    validation_emotion_hash = {}
    training_emotion_hash = {}
    validation_hash = "validation_emotion_hash_pickle"
    training_hash = "training_emotion_hash_pickle"

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--build_hash", help="build train/valid {emotion:file} hash", action="store_true")
    parser.add_argument("-o", "--organize", help="organize affectnet dataset: training_set_imgs/validation_set_imgs")
    # parser.add_argument()
    args = parser.parse_args()
    print("processing dataset . . .")
    # start = time.time()
    start = datetime.datetime.now()
    # print(start)
    db_org = DataOrganizer()

    if (len(sys.argv) <= 1):
        print("** Error: Required Arguments: {-b:--buildhash} or {-o:--organize}")
    # Uses pickle to save emotion_hash
    #       so we don't need to compute the hash everytime
    # emotion_hash = db_org.get_emotion_folders(images_path, csv_training_file)

    # Build {Emotion:List[images]} hash pickles
    if args.build_hash:
        if not os.path.exists(validation_hash):
            print("\nbuilding validation hash . . .")
            validation_emotion_hash = db_org.get_emotion_folders(images_path, csv_validation_file)
            validation_emotion_hash_pickle = open(validation_hash, 'wb')
            pickle.dump(validation_emotion_hash, validation_emotion_hash_pickle)
            validation_emotion_hash_pickle.close()
        else:
            validation_emotion_hash_pickle = open(validation_hash, 'rb')
            validation_emotion_hash = pickle.load(validation_emotion_hash_pickle)
            print("\nloaded validation hash pickle . . .")
            num_total_imgs = 0
            for key, val in validation_emotion_hash.items():
                print("\n\nemotion: ", key)
                print("image list len: ", len(val))
                num_total_imgs += len(val)
            print("Total num validation images: ", num_total_imgs)

        if not os.path.exists(training_hash):
            print("\nbuilding training hash . . .")
            training_emotion_hash = db_org.get_emotion_folders(images_path, csv_training_file)
            training_emotion_hash_pickle = open(training_hash, 'wb')
            pickle.dump(training_emotion_hash, training_emotion_hash_pickle)
            training_emotion_hash_pickle.close()
        else:
            print("\nloading training hash pickle. . .")
            training_emotion_hash_pickle = open(training_hash, 'rb')
            training_emotion_hash = pickle.load(training_emotion_hash_pickle)
            print("\nloaded training hash pickle . . .")
            num_total_training_imgs = 0
            for key, val in training_emotion_hash.items():
                print("\n\nemotion: ", key)
                print("image list len: ", len(val))
                num_total_training_imgs += len(val)
            print("Total num training images: ", num_total_training_imgs)

    # organize training set
    if args.organize:
        total_moved = 0
        move_imgs = 0
        emotion_hash = {}

        # If folder exists: print(already exists)
        if (os.path.exists(args.organize)):
            # Delete the folder
            print(args.organize, " already exists")
        else:
            # organize the image set
            print("\n\norganizing dataset . . .")
            for emotion in emotions:
                print("\norganizing emotion:", emotion)
                # move_imgs = db_org.get_training_set(validation_path, emotion, emotion_hash)
                if (args.organize == "training_set"):
                    training_emotion_hash_pickle = open(training_hash, 'rb')
                    # training_emotion_hash = pickle.load(training_emotion_hash_pickle)
                    emotion_hash = pickle.load(training_emotion_hash_pickle)
                    print("\nloaded training hash pickle . . .")
                    # move_imgs = db_org.get_training_set(args.organize, emotion, training_emotion_hash)
                    # print("Total emotion images moved: ", move_imgs)
                    # total_moved += move_imgs

                elif(args.organize == "validation_set"):
                    validation_emotion_hash_pickle = open(validation_hash, 'rb')
                    # validation_emotion_hash = pickle.load(validation_emotion_hash_pickle)
                    emotion_hash = pickle.load(validation_emotion_hash_pickle)
                    print("\nloaded validation hash pickle . . .")
                    # move_imgs = db_org.get_training_set(args.organize, emotion, validation_emotion_hash)
                    # print("Total emotion images moved: ", move_imgs)
                    # total_moved += move_imgs
                move_imgs = db_org.get_training_set(args.organize, emotion, emotion_hash)
                print("Total emotion images moved: ", move_imgs)
                total_moved += move_imgs

            print("\n\nTotal images moved: ", total_moved)

    end = datetime.datetime.now()
    elapsed = end - start

    print("\n\nstart time: ", start)
    print("end time: ", end)
    print("elapsed time: ", elapsed.seconds, " seconds")
