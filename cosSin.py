import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Создаем обучающий набор данных
x_train = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])  # Входные значения x
y_train = np.array([0.0, 0.8415, 0.9093, 0.1411, -0.7568, -0.9589])  # Ожидаемые выходные значения y

# Определяем архитектуру модели
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))  # Полносвязный слой с 1 входом и 10 нейронами
model.add(Dense(2, activation='softmax'))  # Полносвязный слой с 2 нейронами и функцией активации softmax

# Компилируем модель
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Преобразуем выходные значения в формат one-hot encoding
y_train_categorical = np.zeros((y_train.shape[0], 2))
y_train_categorical[np.arange(y_train.shape[0]), np.where(y_train < 0, 1, 0)] = 1

# Обучение модели
model.fit(x_train, y_train_categorical, epochs=100, batch_size=1)

# Тестирование модели
x_test = np.array([2.0, 3.0, 4.0, np.pi/2, np.pi])
predictions = model.predict(x_test)
predicted_functions = ['sin(x)' if p[0] > p[1] else 'cos(x)' for p in predictions]
print(predicted_functions)
