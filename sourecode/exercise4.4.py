# Exercise 4.4
# Modifikasi LeNet (Example 4.4): jumlah filter diubah
# Conv2D pertama 6 -> 12 filter, Conv2D kedua 16 -> 32 filter
# Perhatikan perubahan Total params pada summary
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten,AveragePooling2D
from keras import optimizers
model = Sequential()
model.add(Conv2D(filters=12, kernel_size=(3, 3), activation='relu',
input_shape=(32,32,1)))                                      # diubah dari 6
model.add(AveragePooling2D())
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))   # diubah dari 16
model.add(AveragePooling2D())
model.add(Flatten())
model.add(Dense(units=120, activation='relu'))
model.add(Dense(units=84, activation='relu'))
model.add(Dense(units=10, activation = 'softmax'))
model.summary()
