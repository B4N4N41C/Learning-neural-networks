from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()
img = Image.open(file_path)

def prcessed_image(img):
    image = img.resize((256, 256), Image.BILINEAR)
    image = np.array(image, dtype=float)
    size = image.shape
    lab = rgb2lab(1.0 / 255 * image)
    X, Y = lab[:, :, 0], lab[:, :, 1:]

    Y = Y / 128
    X = X.reshape((1, size[0], size[1], 1))
    Y = Y.reshape((1, size[0], size[1], 2))
    return X, Y, size

X, Y, size = prcessed_image(img)

model = Sequential()
model.add(InputLayer(input_shape=(None, None, 1)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))

model.compile(optimizer='adam', loss='mse')
model.fit(x=X, y=Y, batch_size=1, epochs=50)

upl = filedialog.askopenfilename()
img = Image.open(upl)
X, Y, size = prcessed_image(img)

output = model.predict(X)
output *= 128
#Output colorizations
min_vals, max_vals = -128.0, 127.0
ab = np.clip(output[0], min_vals, max_vals)

cur = np.zeros((size[0], size[1], 3))
cur[:, :, 0] = np.clip(X[0][:, :, 0], 0, 100)
cur[:, :, 1:] = ab
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(lab2rgb(cur))
plt.show()

