from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

def build_model(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"]):
    # build model
    model = Sequential()

    model.add(Dense(512,activation="relu", input_shape=(28*28,)))
    model.add(Dropout(0.2))
    model.add(Dense(512,activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation="softmax"))

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)

    return model
