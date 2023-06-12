import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

from src.utils.constants import MODEL_PARAMS


class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y


class MnistDataset:

    def __init__(self):
        self.__x_train = None
        self.__y_train = None
        self.__x_test = None
        self.__y_test = None
        self.__train_gen = None
        self.__test_gen = None

    def upload_mnist(self, normalize=True, dtype="float32"):
        # Upload the MNIST dataset
        mnist = tf.keras.datasets.mnist

        # Split the dataset into train and test
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        if normalize:
            # Normalize the dataset
            X_train = X_train / 255.0
            X_test = X_test / 255.0

        # Add a channels dimension
        X_train = X_train[..., tf.newaxis].astype(dtype)
        X_test = X_test[..., tf.newaxis].astype(dtype)

        self.__x_train, self.__y_train = X_train, y_train
        self.__x_test, self.__y_test = X_test, y_test

        self.__train_gen = DataGenerator(self.__x_train, self.__y_train, MODEL_PARAMS.BATCH_SIZE)
        self.__test_gen = DataGenerator(self.__x_test, self.__y_test, MODEL_PARAMS.BATCH_SIZE)

    @property
    def mnist_numpy(self):
        return (self.__x_train, self.__y_train), (self.__x_test, self.__y_test)

    @property
    def mnist_generator(self):
        return self.__train_gen, self.__test_gen

    @staticmethod
    def convert_generator(x, y):
        return DataGenerator(x, y, MODEL_PARAMS.BATCH_SIZE)
