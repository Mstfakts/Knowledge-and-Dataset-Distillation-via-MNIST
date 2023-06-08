import numpy as np


def average_images(X_train, Y_train, num_samples):
    """
    Calculate the average images for each class based on the given training data.

    :param X_train: Training images. The shape of each image should be (28, 28).
    :param Y_train: Training labels corresponding to the images.
    :param num_samples: Number of samples per class to consider for averaging.
    :return: A tuple containing the average images and their corresponding labels.
             The average images are reshaped to have a shape of (10 * num_samples, 28, 28, 1),
             and the labels have a shape of (10 * num_samples,).
    """
    # Reshape the input data
    X_train = X_train.reshape(-1, 28 * 28)

    # separate classes
    class_samples = []
    for i in range(10):
        class_indices = np.where(Y_train == i)
        class_samples.append(X_train[class_indices])

    # avg of every class in the number of samples wanted
    labels = []
    train = []
    for i in range(10):
        data = class_samples[i]
        num = int(data.shape[0] / num_samples)
        for j in range(num_samples):
            class_average = np.mean(data[j * num:(j + 1) * num], axis=0)
            train.append(class_average)
            labels.append(i)

    npa = np.asarray(train, dtype=np.float32)
    train = npa.reshape(10 * num_samples, 28, 28, 1)
    labels = np.asarray(labels, dtype=np.float32)
    labels = labels.reshape(10 * num_samples, )

    return train, labels
