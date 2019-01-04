import numpy as np
from keras.datasets import mnist

from utils.preprocess import preprocess
from models.fully_connected_nn import build_model

if __name__ == '__main__':
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
    print("loss for test data:", score[0])
    print("accuracy for test data:", score[1])
