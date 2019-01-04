import keras

def preprocess(X_train, y_train, X_test, y_test, num_classes):
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
