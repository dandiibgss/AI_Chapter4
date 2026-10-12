# Exercise 4.6
# Modifikasi Example 4.7: tambah Conv2D, MaxPooling, dan Dropout
# sebelum Flatten. Bandingkan jumlah parameter (asli: 591,042)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
# Create model
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1),
activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu'))   # layer tambahan
model.add(MaxPooling2D())                          # layer tambahan
model.add(Dropout(0.2))                            # layer tambahan
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam',
metrics=['accuracy'])
print(model.summary())
