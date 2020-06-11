import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense
import numpy as np

# loading the dataset
x_train = np.load("C:/Users/agarw/Desktop/digit/x_train.npy")
y_train = np.load("C:/Users/agarw/Desktop/digit/y_train.npy")
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

num_pixels = 784
num_classes = 10


# creating model
model = Sequential()
model.add(Dense(num_pixels, input_dim=num_pixels,
                kernel_initializer="normal", activation="relu"))
model.add(Dense(num_classes, kernel_initializer="normal", activation="softmax"))


# compiling the model
model.compile(loss="categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])

# training the data
model.fit(x_train, y_train, batch_size=200, epochs=100, verbose=2)


# saving the model
dig = model.to_json()
with open("C:/Users/agarw/Desktop/digit/dig.json", "w")as json_file:
    json_file.write(dig)
model.save_weights("C:/Users/agarw/Desktop/digit/dig.h5")
print("model saved")
