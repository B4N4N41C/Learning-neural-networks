import time
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.sin(x) + 0.5 * x
    # return x * x - 5 * x + 5


def df(x):
    return np.cos(x) + 0.5
    # return 2 * x - 5


N = 20  # число итераций
xx = 2.5  # начальное значение
lmd = 0.1  # шаг сходимости

x_plt = np.arange(-5.0, 5.0, 0.1)
f_plt = [f(x) for x in x_plt]

plt.ion()  # Включение интерактивного режима отображения графиков
fig, ax = plt.subplots()  # Создание окна и осей для графика
ax.grid(True)  # Отображене сетки на графике

ax.plot(x_plt, f_plt)  # Отображение параболы
point = ax.scatter(xx, f(xx), c='red')  # Отображение точки красным цветом

mn = 100
for i in range(N):
    lmd = 1 / min(i + 1, mn)
    xx = xx - lmd * np.sign(df(xx))  # Изменение аргумента на текущей итерации

    point.set_offsets([xx, f(xx)])  # Отображение нового положения точки

    # Перерисовка графика и задержка на 20 мс
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.2)

plt.ioff()
print(xx)
ax.scatter(xx, f(xx), c='blue')
plt.show()
