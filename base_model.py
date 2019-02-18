from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras import backend

# SmallerVGGNet Model
class BaseModel:
    def build(width, height, depth, classes, finalAct="softmax"):
        model = Sequential()
        in_shape = (height, width, depth)
        chan_dim = -1

        if backend.image_data_format() == "channels_first":
            in_shape = (depth, height, width)
            chan_dim = 1

        model.add(Conv2D(64, (3,3), input_shape=in_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(3,3))) # to prevent over-fitting
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation(finalAct))

        # # CONV -> RELU -> POOL
        # model.add(Conv2D(32, (3, 3), padding="same", input_shape=in_shape))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chan_dim))
        # model.add(MaxPooling2D(pool_size=(3, 3)))
        # model.add(Dropout(0.25))
#
        # # (CONV -> RELU) * 2 -> POOL
        # model.add(Conv2D(64, (3, 3), padding="same"))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chan_dim))
        # model.add(Conv2D(64, (3,3 ), padding="same"))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chan_dim))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
        #
        # # (CONV -> RELU) * 2 -> POOL
        # model.add(Conv2D(128, (3, 3), padding="same"))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chan_dim))
        # model.add(Conv2D(128, (3,3 ), padding="same"))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chan_dim))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
        #
        # # FC -> RELU
        # # Fully connected layers specified by Dense
        # model.add(Flatten())
        # model.add(Dense(1024))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.5))
        #
        # # use a *softmax* activation for single-label classification
		# # and *sigmoid* activation for multi-label classification
        # model.add(Dense(classes))
        # model.add(Activation(finalAct))

        # opt = Adam(lr=0.5, decay=0.5/3)
        # Configure learning process via the compile method
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        return model
