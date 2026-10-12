# Exercise 4.15
# Modifikasi kode Part 4: n = 1 -> n = 2 (layer konvolusi ke-2 = block1_conv2)
# Jalankan di Google Colab
from keras.applications.vgg19 import VGG19
from matplotlib import pyplot

model = VGG19()

# retrieve weights from the second convolutional layer
n = 2          # <-- diubah dari n = 1
filters, biases = model.layers[n].get_weights()
s = filters.shape
print("Layer:", model.layers[n].name)
print("Color channels: ", s[0])
print("Filter size: ", s[1], s[2])
print("Total number of filters : ", s[3])

# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# plot first few filters
n_filters, ix = 4, 1

pyplot.figure(figsize=(10, 10))
for i in range(n_filters):
    # get the filter
    f = filters[:, :, :, i]
    # plot each channel separately
    for j in range(s[0]):
        ax = pyplot.subplot(n_filters, s[0], ix)
        ax.set_xticks([])
        ax.set_yticks([])
        pyplot.imshow(f[j, :, :], cmap='gray')
        ix += 1
pyplot.show()