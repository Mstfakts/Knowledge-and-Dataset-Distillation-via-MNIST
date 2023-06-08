import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D


class TeacherModel(Model):
    def __init__(self, T: float):
        super(TeacherModel, self).__init__()

        self.T = T

        self.conv1 = Conv2D(32, 3, activation="relu")
        self.flatten = Flatten()

        self.d1 = Dense(1200, activation="relu")
        self.d2 = Dense(1200, activation="relu")
        self.d3 = Dense(10)

        self.dropout_layer_hidden = tf.keras.layers.Dropout(rate=0.5)

        self.output_layer = tf.keras.layers.Softmax()

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)

        x = self.d1(x)
        x = self.dropout_layer_hidden(x)

        x = self.d2(x)
        x = self.dropout_layer_hidden(x)

        x = self.d3(x)
        x = self.output_layer(x / self.T)
        return x
