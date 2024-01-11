import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split

# %matplotlib inline

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Станартизация входных данных
x_train = x_train / 255
x_test = x_test / 255

# Преобразование входных данных
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

limit = 5000
x_train_data = x_train[:limit]
y_train_data = y_train_cat[:limit]

x_valid = x_train[limit:limit*2]
y_valid = y_train_cat[limit:limit*2]

# # отображение первых 25 изображений из обучающей выборки
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(x_train[i], cmap=plt.cm.binary)
# plt.show()

model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(300, activation='relu'),
    # BatchNormalization(),
    Dense(10, activation='softmax')
])

# print(model.summary())  # вывод структуры нейронной сети

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x_train, y_train_cat, test_size=0.2)

his = model.fit(x_train_data, y_train_data, batch_size=32, epochs=50, validation_data=(x_valid, y_valid))
# model.evaluate(x_test, y_test_cat)

plt.plot(his.history['loss'])
plt.plot(his.history['val_loss'])
plt.show()


# n = 2
# x = np.expand_dims(x_test[n], axis=0)
# res = model.predict(x)
# print(res)
# print(f"Распознанное изображение: {np.argmax(res)}")
#
# plt.imshow(x_test[n], cmap=plt.cm.binary)
# plt.show()
#
# pred = model.predict(x_test)
# pred = np.argmax(pred, axis=1)
#
# print(pred.shape)
#
# print(pred[:20])
# print(y_test[:20])
#
# mask = y_test == pred
# print(mask[:10])
#
# x_false = x_test[~mask]
# p_false = pred[~mask]
#
# print(x_false.shape)
#
# for i in range(5):
#     print(f"Распознанное изображение: {p_false[i]}")
#     plt.imshow(x_false[i], cmap=plt.cm.binary)
#     plt.show()
