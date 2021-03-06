import numpy as np
import json
from keras.datasets import mnist

from utils.preprocess import preprocess
from models.fully_connected_nn import build_model

if __name__ == '__main__':
    # loading parameters
    with open("configs/default.json", mode="r") as f:
        params = json.load(f)

    num_classes = params["num_classes"]
    epochs = params["epochs"]
    batch_size = params["batch_size"]

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
    print("loss for test data:", score[0])
    print("accuracy for test data:", score[1])
