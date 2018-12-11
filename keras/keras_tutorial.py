import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

def preprocess_data(X_train, y_train, X_test, y_test, num_classes):
    # preprocess data
    X_train = X_train.reshape(len(X_train), 28*28)
    X_test = X_test.reshape(len(X_test), 28*28)
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train /= 255 # regularize data
    X_test /= 255

    # convert to categorical vectors
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return X_train, y_train, X_test, y_test

def build_model():
    # build model
    model = Sequential()

    model.add(Dense(512,activation="relu", input_shape=(28*28,)))
    model.add(Dropout(0.2))
    model.add(Dense(512,activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation="softmax"))

    model.compile(loss="categorical_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])

    return model

if __name__ == '__main__':
    # setting parameters
    batch_size = 128
    num_classes = 10
    epochs = 50

    # loading datasets
    (X_train, y_train),(X_test, y_test) = mnist.load_data()

    # preprocess data
    X_train, y_train, X_test, y_test = preprocess_data(X_train, y_train, X_test, y_test, num_classes)

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
