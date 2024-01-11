import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from io import BytesIO
from PIL import Image
from tensorflow.keras.preprocessing import image as kp_image
import tkinter as tk
from tkinter import filedialog

model = keras.applications.VGG16()

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()
if not file_path:
    print('No file selected.')
    exit()

img = Image.open(file_path)
plt.imshow(img)

img = np.array(img)
x = keras.applications.vgg16.preprocess_input(img)
print(x.shape)
x = np.expand_dims(x, axis=0)
print(x.shape)

res = model.predict(x)
print(np.argmax(res))
