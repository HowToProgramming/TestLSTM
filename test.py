import numpy as np

from lstm import LSTM

data = open("japanesenameexample", 'r').read()
unique = list(set(data))
def onehot(char, set_of_char):
    one_huh = [[0]] * len(set_of_char)
    one_huh[set_of_char.index(char)] = [1]
    return one_huh
    
def oneheat(char):
    return np.array(onehot(char, unique))

data = list(map(oneheat, data))
lstm = LSTM(len(unique), len(unique), 128)
for epoch in range(1000):
    loss = 0
    _ = 0
    for i in range(len(data) - 1):
        X = np.array(data[:i + 1])
        Y = data[i+1]
        loss += lstm.backpropagation(X, Y, learning_rate=0.02)
        _ += 1
    print("Epoch / Iteration: {} || Loss: {} / char".format(epoch + 1, loss[0] / _))
    if loss / _ < 0.1:
        break

# test data
X = np.random.choice(unique)
X = [oneheat(X)]
LENGTH = 500
for i in range(LENGTH):
    X.append(lstm.test(np.array(X)))
X = np.array(X).tolist()
def decode(idx):
    return unique[idx.index([1])]
with open("my_name_is.txt", "w+") as wuh:
    wuh.write("".join(list(map(decode, X))))
