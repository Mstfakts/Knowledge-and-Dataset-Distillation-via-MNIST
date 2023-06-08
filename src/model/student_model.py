import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


class StudentModel(Model):
    def __init__(self, T):
        super(StudentModel, self).__init__()

        self.T = T

        self.input_layer = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.d1 = Dense(10, activation="relu")
        self.d2 = Dense(10)
        self.output_layer = tf.keras.layers.Softmax()

    def call(self, x):
        x = self.input_layer(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.output_layer(x / self.T)
        return x
