# Exercise 4.3
# Modifikasi Example 4.3: dense layer 10 neuron -> 5 neuron
# Bandingkan waktu training dan akurasi dengan versi 10 neuron
import time
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras import optimizers
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

whole_data = load_breast_cancer()
X_data = whole_data.data
y_data = whole_data.target
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,
test_size = 0.3, random_state = 7)

features = X_train.shape[1]

#MPL 3-layer Model ==============================================
model = Sequential()
model.add(Dense(5, activation='relu', input_shape=[X_train.shape[1]]))   # diubah dari 10
model.add(Dense(5, activation='relu'))                                    # diubah dari 10
model.add(Dense(1))
print(model.summary())
print(model.get_config())

model.compile(optimizer = 'adam', loss = 'mean_squared_error',
metrics = ['mse'])

start = time.time()
model.fit(X_train, y_train, batch_size = 50, validation_split=0.2,
epochs = 100, verbose = 1)
print('Training time: %.2f seconds' % (time.time() - start))

results = model.evaluate(X_test, y_test)

print('loss: ', results[0])
print('accuracy: ', results[1])
