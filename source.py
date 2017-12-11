import numpy
import requests
import time
from PIL import Image
from io import BytesIO
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils


def load_data():
    # response = requests.get("https://cs.nyu.edu/~roweis/data/olivettifaces.gif")
    # img = Image.open(BytesIO(response.content))
    img = Image.open("./olivettifaces.gif")
    img_ndarray = numpy.asarray(img, dtype='float64')/256
    # total 400 pictures, size 57 * 47 = 2679
    faces = numpy.empty([400, 2679], dtype='float64')
    for row in range(20):
        for col in range(20):
            faces[row*20+col] = numpy.ndarray.flatten(img_ndarray[row*57:(row+1)*57,
                                                      col*47: (col+1)*47])

    labels = numpy.empty(400, dtype='int64')
    for i in range(40):
        labels[i*10:(i+1)*10] = i

    train_data = numpy.empty((320, 2679), dtype=numpy.float64)
    train_label = numpy.empty(320, dtype=numpy.int64)
    valid_data = numpy.empty((40, 2679), dtype=numpy.float64)
    valid_label = numpy.empty(40, dtype=numpy.int64)
    test_data = numpy.empty((40, 2679), dtype=numpy.float64)
    test_label = numpy.empty(40, dtype=numpy.int64)

    for i in range(40):
        train_data[i*8:(i+1)*8] = faces[10*i: 10*i+8]
        train_label[i*8: (i+1)*8] = labels[10*i: 10*i+8]
        test_data[i] = faces[10*i+8]
        test_label[i] = labels[10*i+8]
        valid_data[i] = faces[10*i+9]
        valid_label[i] = labels[10*i+9]

    rval = [(train_data, train_label), (test_data, test_label),
            (valid_data, valid_label)]
    return rval


def Net_Model(lr=0.01, decay=1e-6, momentum=0.9):
    model = Sequential()
    model.add(Conv2D(5, (3, 3), strides=(1, 1), activation='relu', input_shape=(57, 47, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(10, (3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(40, activation='softmax'))

    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model


def train_model(model, x_train, y_train, x_val, y_val):
    model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=40, epochs=40)
    model.save_weights('model_weight.h5', overwrite=True)
    return model


def test_model(model, x, y):
    scores = model.evaluate(x, y, verbose=0)
    return scores


if __name__ == "__main__":
    print("Hello, World")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()
    X_train = x_train.reshape(x_train.shape[0], 57, 47, 1)
    X_val = x_val.reshape(x_val.shape[0], 57, 47, 1)
    X_test = x_test.reshape(x_test.shape[0], 57, 47, 1)

    Y_train = np_utils.to_categorical(y_train, 40)
    Y_test = np_utils.to_categorical(y_test, 40)
    Y_val = np_utils.to_categorical(y_val, 40)

    model = Net_Model()
    model.summary()
    train_model(model, X_train, Y_train, X_val, Y_val)
    score = test_model(model, X_test, Y_test)

    print()
    classes_train = model.predict_classes(X_val)
    print()
    print("Train Period => ")
    print(classes_train)
    print(y_val)
    test_accuracy = numpy.mean(numpy.equal(y_val, classes_train))
    print("accuracy: ", test_accuracy)
    print()
    print("Test Period =>git ")
    classes = model.predict_classes(X_test)
    print(classes)
    print(y_test)
    test_accuracy = numpy.mean(numpy.equal(y_test, classes))
    print("accuracy: ", test_accuracy)
