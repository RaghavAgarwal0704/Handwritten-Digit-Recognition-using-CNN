import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.utils import np_utils


(x_train, y_train), (x_test, y_test) = mnist.load_data()


num_pixels = x_train.shape[1]*x_train.shape[2]
x_train = x_train.reshape((x_train.shape[0], num_pixels)).astype("float32")
x_test = x_test.reshape((x_test.shape[0], num_pixels)).astype("float32")


x_test /= 255
x_train /= 255


y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

np.save("C:/Users/agarw/Desktop/digit/x_train", x_train)
np.save("C:/Users/agarw/Desktop/digit/y_train", y_train)
np.save("C:/Users/agarw/Desktop/digit/x_test", x_test)
np.save("C:/Users/agarw/Desktop/digit/y_test", y_test)
