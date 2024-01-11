import numpy as np


def f(x):
    return 2 / (1 + np.exp(-x)) - 1


def df(x):
    return 0.5 * (1 + x) * (1 - x)


W1 = np.array([[-0.2, 0.3, -0.4], [0.1, -0.3, -0.4]])
W2 = np.array([0.2, 0.3])


def go_forward(inp):
    sum = np.dot(W1, inp)
    out = np.array([f(x) for x in sum])

    sum = np.dot(W2, out)
    y = f(sum)
    return (y, out)


def train(epoch):
    global W1, W2
    lmd = 0.01  # Шаг обучения
    N = 10000  # Число итераций при обучении
    count = len(epoch)
    for k in range(N):
        x = epoch[np.random.randint(0, count)]  # Случайный выбор входного сигнала из обучающей выборки
        y, out = go_forward(x[0:3])  # Прямой проход по НС и вычисление выходных значений нейросети
        e = y - x[-1]  # Ошибка
        delta = e * df(y)  # Локальный градиент
        W2[0] = W2[0] - lmd * delta * out[0]  # Корректировка веса первой связи
        W2[1] = W2[1] - lmd * delta * out[1]  # Корректировака веса второй связи

        delta2 = W2 * delta * df(out)  # Вектор из 2-х величин локальных градиентов

        # Корректировка связей первого слоя
        W1[0, :] = W1[0, :] - np.array(x[0:3]) * delta2[0] * lmd
        W1[1, :] = W1[1, :] - np.array(x[0:3]) * delta2[1] * lmd


epoch = [(-1, -1, -1, -1),
         (-1, -1, 1, 1),
         (-1, 1, -1, -1),
         (-1, 1, 1, 1),
         (1, -1, -1, -1),
         (1, -1, 1, 1),
         (1, 1, -1, -1),
         (1, 1, 1, -1)]
train(epoch)  # Запуск обучения сети

# Проверка полученных результатов
for x in epoch:
    y, out = go_forward(x[0:3])
    print(f"Выходные значения НС: {y} => {x[-1]}")
