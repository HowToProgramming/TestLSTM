import numpy as np

from lstm import LSTM

data = "never gonna give you up never gonna let you down never gonna run around and desert you"
unique = list(set(data))
def onehot(char, set_of_char):
    one_huh = [[0]] * len(set_of_char)
    one_huh[set_of_char.index(char)] = [1]
    return one_huh

def oneheat(char):
    return np.array(onehot(char, unique))

data = list(map(oneheat, data))
lstm = LSTM(len(unique), len(unique), 128)
for epoch in range(69420):
    loss = 0
    for i in range(len(data) - 1):
        X = np.array(data[:i + 1])
        Y = data[i+1]
        loss += lstm.backpropagation(X, Y)
    print(epoch + 1, loss[0])