# Exercise 4.10
# Modifikasi Example 4.11: klasifikasi dengan VGG19 dan bandingkan dengan VGG16
import time
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import vgg16, vgg19
from keras.applications.imagenet_utils import decode_predictions
import numpy as np

import os
img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Elephant.jpg')
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

for name, module, net in [("VGG16", vgg16, vgg16.VGG16),
                          ("VGG19", vgg19, vgg19.VGG19)]:
    model = net(weights='imagenet')
    processed = module.preprocess_input(x.copy())

    # prediction
    start = time.time()
    predictions = model.predict(processed)
    t = time.time() - start
    results = decode_predictions(predictions)

    print(name, "| params:", model.count_params(),
          "| waktu prediksi: %.3f detik" % t)
    print(results)
