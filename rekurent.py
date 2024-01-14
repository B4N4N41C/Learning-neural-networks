import os
import numpy as np
import re

from tensorflow.keras.layers import Dense, SimpleRNN, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with open('train_data_true.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    text = text.replace('\ufeff', '')  # удаляем первый символ
    text = re.sub(r'[^А-я] ', '', text)  # заменяем все символы кроме русских букв на пустую строку

# парсим текст, как последовательность символов
num_charctrs = 34 # 33 буквы + пробел
tokenizer = Tokenizer(num_words=num_charctrs, char_level=True) # токенизируем на уровне символов
tokenizer.fit_on_texts([text]) # формируем токены на основе частности в нашем тексте
print(tokenizer.word_index)

inp_chars = 6
data = tokenizer.texts_to_matrix(text) # преобразуем исходный текст в массив OHE
n = data.shape[0] - inp_chars

X = np.array([data[i: i + inp_chars] for i in range(n)])
Y = data[inp_chars:] # передсказание следующего символа

print(data.shape)

model = Sequential()
model.add(Input((inp_chars, num_charctrs))) # при тренировке в реккурентные модели keras подаётся сразу вся последовательность символов
model.add(SimpleRNN(128, activation='tanh')) # рекурентный слой на 128 нейронов
model.add(Dense(num_charctrs, activation='softmax')) # полносвязный слой с softmax
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

history = model.fit(X, Y, batch_size=32, epochs=100)


def buildPhrase(inp_str, str_len=50):
    for i in range(str_len):
        x = []
        for j in range(i, i + inp_chars):
            x.append(tokenizer.texts_to_matrix(inp_str[j])) # пребращуем символ в One Hot Encoding)
        x = np.array(x)
        inp = x.reshape(1, inp_chars, num_charctrs)

        pred = model.predict(inp) # предсказываем OHE четвёртого символа
        d = tokenizer.index_word[pred.argmax(axis=1)[0]] # получаем ответ в сивольном представлении

        inp_str += d # дописываем строку

    return inp_str

res = buildPhrase('утренн')
print(res)

