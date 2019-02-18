from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from sklearn import preprocessing
from base_model import BaseModel

import pickle

class TrainModel(BaseModel):
    def __int__(self):
        self.base_model = BaseModel

    def load_train_pickles(self):
        train_pickle = "train_data_pickle"
        train_label_pickle = "train_label_pickle"

        # load data
        train_data_pickle_load = open(train_pickle, 'rb')
        train_data = pickle.load(train_data_pickle_load)
        print("\n\n# train data pickle entries: ", len(train_data))

        # load labels
        train_label_pickle_load = open(train_label_pickle, 'rb')
        train_labels = pickle.load(train_label_pickle_load)
        print("# train label pickle entries: ", len(train_labels))

        # return train_data, train_labels
        return train_data, train_labels
        # pass

    def load_valid_pickles(self):
        valid_pickle = "valid_data_pickle"
        valid_label_pickle = "valid_label_pickle"

        valid_data_pickle_load = open(valid_pickle, 'rb')
        valid_data = pickle.load(valid_data_pickle_load)
        print("\n# valid data pickle entries: ", len(valid_data))

        # valid_label_pickle = "valid_label_pickle"
        valid_label_pickle_load = open(valid_label_pickle, 'rb')
        valid_labels = pickle.load(valid_label_pickle_load)
        print("# valid label pickle entries: ", len(valid_labels))

        return valid_data, valid_labels
        # pass

    def one_hot_encode(self, labels):
        encoded_labels = to_categorical(labels)
        return encoded_labels
        # pass

if __name__ == "__main__":
    width = 100
    height = 100
    depth = 3
    classes = 8

    trainer = TrainModel()

    # Initiate Model
    model = BaseModel.build(width,height,depth,classes)

    # Load training data and labels
    train_data, train_labels = trainer.load_train_pickles()
    valid_data, valid_labels = trainer.load_valid_pickles()

    # encode labels
    label_binner = preprocessing.LabelBinarizer()
    label_binner.fit(train_labels)
    print(label_binner.classes_)
    # label_binner.transform(["anger"])
    encoded_labels = label_binner.transform(train_labels)

    # for encoded_label in encoded_labels: print(encoded_label)
    # encoded_labels = model.one_hot_encode(train_labels)

    # model.build(width, height, depth, classes)
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    emotions = ["anger", "contempt", "disgust", "fear", "happiness",
            "sadness", "surprise", "uncertain"]
    # train_batches = ImageDataGenerator().flow_from_directory("training_set_imgs", target_size = (100,100), classes = emotions, batch_size=10)
    # valid_batches = ImageDataGenerator().flow_from_directory("validation_set_imgs", target_size = (100,100), classes = emotions, batch_size=10)
    train_batches = ImageDataGenerator().flow_from_directory("training_set_imgs", target_size = (100,100), classes = emotions, batch_size=202)
    valid_batches = ImageDataGenerator().flow_from_directory("validation_set_imgs", target_size = (100,100), classes = emotions, batch_size=55)


    # aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                    # height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                    # horizontal_flip=True, fill_mode="nearest")
    # training the model !!
    print("training model...")
    model.fit_generator(train_batches, steps_per_epoch=5, validation_data=valid_batches, validation_steps=3, epochs=25, verbose=2)
    # model.fit_generator(train_batches, steps_per_epoch=1111, validation_data=valid_batches, validation_steps=100, epochs=25, verbose=2)
    # model.fit_generator(generator=(train_data, train_labels), steps_per_epoch=5, validation_data=(valid_data,valid_labels), validation_steps=3, epochs=25, verbose=2)


#
# The factor pairs of 224,422 are:
# 1 × 224422 = 224,422
# 2 × 112211 = 224,422
# 11 × 20402 = 224,422
# 22 × 10201 = 224,422
# 101 × 2222 = 224,422
# 202 × 1111 = 224,422

# 224,422 images
# batches of size 202 or 1111
# steps per epoch 1111 or 202

# 5500 images
# batches of 55
# steps per epoch 100
