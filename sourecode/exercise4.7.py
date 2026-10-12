# Exercise 4.7
# Modifikasi Example 4.9:
#  (a) jumlah filter Conv2D 32 -> 64
#  (b) ukuran filter (5,5) -> (3,3)
# Ketiga konfigurasi (asli, a, b) dijalankan dan dibandingkan performanya
import time
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][width][height][channels]
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

def build_model(n_filters, kernel):
    model = Sequential()
    model.add(Conv2D(n_filters, kernel, input_shape=(28, 28, 1),
    activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
    metrics=['accuracy'])
    return model

configs = [("Asli: 32 filter, (5,5)", 32, (5, 5)),
           ("(a) 64 filter, (5,5)", 64, (5, 5)),
           ("(b) 32 filter, (3,3)", 32, (3, 3))]

for name, n_filters, kernel in configs:
    model = build_model(n_filters, kernel)
    start = time.time()
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
    epochs=10, batch_size=200)
    t = time.time() - start
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(name, "| params:", model.count_params(),
          "| CNN Error: %.2f%%" % (100-scores[1]*100),
          "| waktu: %.1f detik" % t)
