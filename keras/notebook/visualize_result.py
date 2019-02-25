# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt

import sys, os
sys.path.append('..')

from utils.preprocess import preprocess
from models.fully_connected_nn import build_model
# -

print(sys.path)

# +
# setting parameters
batch_size = 128
num_classes = 10
epochs = 1

# loading datasets
(X_train, y_train),(X_test, y_test) = mnist.load_data()

# preprocess data
X_train, y_train, X_test, y_test = preprocess(X_train, y_train, X_test, y_test, num_classes)

# split train & validation data
X_train, X_valid = np.split(X_train, [50000])
y_train, y_valid = np.split(y_train, [50000])

# build model
model = build_model()

# fitting model to training data
fit = model.fit(X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(X_valid,y_valid))

# evaluate model
score = model.evaluate(X_test, y_test, verbose=0)
predict_classes = model.predict_classes(X_test)
print("loss for test data:", score[0])
print("accuracy for test data:", score[1])
# -

num = 10
fig = plt.figure(figsize=(15, 25))
for i in range(num):
    ax = fig.add_subplot(1,num,i+1, xticks=[], yticks=[])
    ax.set_title("predict:{0}({1})".format(predict_classes[i], np.argmax(y_test[i])))
    ax.imshow(X_test[i].reshape((28, 28)), cmap='gray')
