# Exercise 4.16
# Modifikasi kode Part 4: n_filters, ix = 4, 1 -> n_filters, ix = s[3], 1
# (tampilkan semua filter pada layer 1)
from keras.applications.vgg19 import VGG19
from matplotlib import pyplot

model = VGG19()

n = 1
filters, biases = model.layers[n].get_weights()
s = filters.shape
print("Layer:", model.layers[n].name)
print("Color channels: ", s[0])
print("Filter size: ", s[1], s[2])
print("Total number of filters : ", s[3])

f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

n_filters, ix = s[3], 1      # <-- diubah dari n_filters, ix = 4, 1

pyplot.figure(figsize=(10, 10))
for i in range(n_filters):
    f = filters[:, :, :, i]
    for j in range(s[0]):
        ax = pyplot.subplot(n_filters, s[0], ix)
        ax.set_xticks([])
        ax.set_yticks([])
        pyplot.imshow(f[j, :, :], cmap='gray')
        ix += 1
pyplot.show()