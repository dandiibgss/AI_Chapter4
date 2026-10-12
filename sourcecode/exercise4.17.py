# Exercise 4.17
# Modifikasi kode Part 6: n = 1 -> n = 2 (lalu coba n = 4, 5, dst.)
# Butuh file Elephant.jpg di folder yang sama (upload ke Colab)
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
import matplotlib.pyplot as plt
from numpy import expand_dims

# fungsi dari Part 5 (harus didefinisikan dulu)
def plot_feature_maps(feature_maps):
    # plot all feature maps
    col = 8
    row = int(feature_maps.shape[3] / col)
    ix = 1
    plt.figure(figsize=(20, 20))
    for _ in range(row):
        for _ in range(col):
            ax = plt.subplot(row, col, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
            ix += 1
    plt.show()

# load the model
model = VGG19()
# Select the hidden layer to visualize
n = 2          # <-- diubah dari n = 1 (coba juga 4, 5, ...)
# redefine model to output right after the hidden layer
model = Model(inputs=model.inputs, outputs=model.layers[n].output)
model.summary()

# load the image with the required shape
img = load_img(r'D:\KULIAH\SEMESTER 5\AI\chapter4\sourecode\Elephant.jpg', target_size=(224, 224))
img = img_to_array(img)
img = expand_dims(img, axis=0)
img = preprocess_input(img)

# get feature maps of the selected layer
feature_maps = model.predict(img)
print("Feature maps: ", feature_maps.shape)
plot_feature_maps(feature_maps)