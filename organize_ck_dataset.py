from pathlib import Path
import os, os.path
import shutil


"""
Emotion Codes:
    0   Neutral
    1   Anger
    2   Contempt
    3   Disgust
    4   Fear
    5   Happy
    6   Sadness
    7   Surprise
"""
# Emotion Labels
# \face-rec\Emotion_labels\Emotion

# Sequence Images
# \face-rec\extended-cohn-kanade-images\cohn-kanade-images

class CategorizeImages(object):
    def get_category_images(self, path):
        folders = []
        files = []
        # neutral = anger = contempt = disgust = fear = happy = sadness = surprise = []
        # emotion_list = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
        emotion_hash = {"neutral": [],"anger": [],"contempt": [], "disgust": [], "fear": [], "happy": [], "sadness": [], "surprise": []}
        for dir in os.scandir(path):
            for image in os.scandir(dir.path):
                for emotion in os.scandir(image.path):
                    if emotion.is_dir():
                        folders.append(emotion.path)
                    elif emotion.is_file():
                        files.append(emotion.name)
                        file = open(image.path + "\\" + emotion.name, "r")
                        emotion_code = file.read()
                        # print("emotion code: ", float(emotion_code), "\nfilename: ", emotion.name)
                        # print()
                        if float(emotion_code) == 0.0:
                            emotion_hash["neutral"].append(emotion.name)
                            # neutral.append(emotion.name)
                        elif float(emotion_code) == 1.0:
                            emotion_hash["anger"].append(emotion.name)
                            # anger.append(emotion.name)
                        elif float(emotion_code) == 2.0:
                            emotion_hash["contempt"].append(emotion.name)
                            # contempt.append(emotion.name)
                        elif float(emotion_code) == 3.0:
                            emotion_hash["disgust"].append(emotion.name)
                            # disgust.append(emotion.name)
                        elif float(emotion_code) == 4.0:
                            emotion_hash["fear"].append(emotion.name)
                            # fear.append(emotion.name)
                        elif float(emotion_code) == 5.0:
                            emotion_hash["happy"].append(emotion.name)
                            # happy.append(emotion.name)
                        elif float(emotion_code) == 6.0:
                            emotion_hash["sadness"].append(emotion.name)
                            # sadness.append(emotion.name)
                        elif float(emotion_code) == 7.0:
                            emotion_hash["surprise"].append(emotion.name)
                            # surprise.append(emotion.name)
        return folders, files, emotion_hash

    def categorize_images(self, path, emotion, emotion_hash):
        emotion_img_list = emotion_hash[emotion]
        # print("\nemotion_img_list: ", emotion_img_list)
        emotion_folder = os.getcwd() + "\\" + emotion
        # print("\nemotion folder: ", emotion_folder)
        # print("\ncurrent directory: ", os.getcwd())
        path_dest = emotion

        if not os.path.exists(emotion):
            os.makedirs(emotion)

        for img in emotion_img_list:
            temp = img.split("_")

            path_src = "{}\\{}\\{}\\{}_{}_{}.png".format(path, temp[0], temp[1], temp[0], temp[1], temp[2])

            dest_check_img_exist = "{}\\{}_{}_{}.png".format(emotion, temp[0], temp[1], temp[2])
            img_path_exists = Path(dest_check_img_exist)

            if not img_path_exists.exists():
                shutil.copy(path_src, path_dest)


if __name__ == "__main__":
    emotion_label_path = "Emotion_labels\\Emotion"
    images_dest = "extended-cohn-kanade-images\\cohn-kanade-images"
    emotion_list = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]

    ImageCategorizer = CategorizeImages()
    folders, files, emotion_hash = ImageCategorizer.get_category_images(emotion_label_path)

    total_labeled_images = 0
    for emotion in emotion_list:
        total_labeled_images += len(emotion_hash[emotion])
        print("\nlen of ", emotion, ": ", len(emotion_hash[emotion]))

    print("\n total labeled images: ", total_labeled_images)

    for emotion in emotion_list:
        ImageCategorizer.categorize_images(images_dest, emotion, emotion_hash)
