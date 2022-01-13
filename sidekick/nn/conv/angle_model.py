from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, Dropout, MaxPool2D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow import nn as tfn
import tensorflow.keras.backend as K

class MiniVgg:
    @staticmethod
    def build(width,height,depth,classes):
        model=Sequential()
        inputShape=(height,width,depth)
        chanDim=-1

        if K.image_data_format()=="channel_first":
            inputShape=(depth,height,width)
            chanDim=1

        model.add(Conv2D(32,(5,5),input_shape=inputShape))
        model.add(Activation(tfn.relu))
        model.add(BatchNormalization(chanDim))

        model.add(Conv2D(32, (5, 5)))
        model.add(Activation(tfn.relu))
        model.add(BatchNormalization(chanDim))

        model.add(Conv2D(32, (5, 5)))
        model.add(Activation(tfn.relu))
        model.add(BatchNormalization(chanDim))

        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
#-----------------------------------#
        model.add(Conv2D(32, (5, 5), input_shape=inputShape))
        model.add(Activation(tfn.relu))
        model.add(BatchNormalization(chanDim))

        model.add(Conv2D(32, (5, 5)))
        model.add(Activation(tfn.relu))
        model.add(BatchNormalization(chanDim))

        model.add(Conv2D(64, (5, 5)))
        model.add(Activation(tfn.relu))
        model.add(BatchNormalization(chanDim))

        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

#-----------------------------#

        model.add(Conv2D(64, (5, 5), input_shape=inputShape))
        model.add(Activation(tfn.relu))
        model.add(BatchNormalization(chanDim))

        model.add(Conv2D(64, (5, 5)))
        model.add(Activation(tfn.relu))
        model.add(BatchNormalization(chanDim))

        model.add(Conv2D(64, (5, 5)))
        model.add(Activation(tfn.relu))
        model.add(BatchNormalization(chanDim))

        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        #-----------------------------#

        model.add(Conv2D(64, (5, 5)))
        model.add(Activation(tfn.relu))
        model.add(BatchNormalization(chanDim))
        model.add(Conv2D(64, (5, 5)))
        model.add(Activation(tfn.relu))
        model.add(BatchNormalization(chanDim))

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation(tfn.relu))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(classes))
        model.add(Activation(tfn.softmax))

        return model
