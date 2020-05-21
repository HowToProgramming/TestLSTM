import numpy as np
from numpy.random import randn as build_weight

class dW_dB(object):
    """using for backpropagation on most gate\n
    and why I don't want to fix the weight_bias class below, because I'm lazy, that's all"""
    def __init__(self, WxShape, WhShape, Bshape):
        self.dwx = np.zeros(WxShape)
        self.dwh = np.zeros(WhShape)
        self.db = np.zeros(Bshape)
    
    def __add__(self, xhb: tuple):
        """
        xhb is a tuple
        - 0: dwx, 1: dwh, 2: db"""
        new = self
        new.dwx += xhb[0]
        new.dwh += xhb[1]
        new.db += xhb[2]
        return new

class weight_bias(object):
    """Using for forget gate, input gate, input modulation gate and output gate\n
    to not waste variables and make things easier"""
    def __init__(self, input_size, hidden_size):
        self.xh = build_weight(hidden_size, input_size)
        self.hh = build_weight(hidden_size, hidden_size)
        self.bh = np.zeros((hidden_size, 1))
    
    def process(self, x, h, f):
        """return the result of processing, aka f(W_xh @ x + W_hh @ h + b)
        - f: activation function, in LSTM, we use tanh and sigmoid function
        - x: input layer
        - h: hidden layer"""
        return f(self.xh @ x + self.hh @ h + self.bh)
    
    def __sub__(self, dw: dW_dB):
        new = self
        new.xh -= dw.dwx
        new.hh -= dw.dwh
        new.bh -= dw.db
        return new

    
class LSTM:
    """LSTM Many-to-one from numpy, inspired from Victorzhou's RNN, https://victorzhou.com/blog/intro-to-rnns/"""
    def __init__(self, input_size, output_size, hidden_size):
        self.forget = weight_bias(input_size, hidden_size)
        self.input = weight_bias(input_size, hidden_size)
        self.inputmodulation = weight_bias(input_size, hidden_size)
        self.output = weight_bias(input_size, hidden_size)
        self.why = build_weight(output_size, hidden_size)
        self.by = np.zeros((output_size, 1))
        self.hiddensize = hidden_size
    
    @staticmethod
    def sigmoid(x):
        """sigmoid(x) = 1 / (1 + e^(-x)), used for forget gate, input gate and output gate"""
        return 1 / (1 + np.exp(-x))
    
    tanh = np.tanh

    @staticmethod
    def dsigmoid_dx(out):
        """dsigmoid/dx = sigmoid * (1 - sigmoid), used for backpropagation"""
        return out * (1 - out)
    
    @staticmethod
    def dtanh_dx(out):
        """dtanh/dx = 1 - tanh^2, used for backpropagation"""
        return 1 - out ** 2

    def forward(self, inp):
        """will write desc later, this is freaking hard to understand"""
        self.cells = {0: np.zeros((self.hiddensize, 1))}
        self.hidden = {0: np.zeros((self.hiddensize, 1))}

        # record datas for backpropagation later
        self.forget_data = dict()
        self.input_mod_data = dict()
        self.input_data = dict()
        self.output_data = dict()
        self.last_input = inp

        # forward
        for idx in range(len(inp)):
            forget_gate = self.forget.process(inp[idx], self.hidden[idx], self.sigmoid)
            input_gate = self.input.process(inp[idx], self.hidden[idx], self.sigmoid)
            input_mod = self.inputmodulation.process(inp[idx], self.hidden[idx], self.tanh)
            next_cell = forget_gate * self.cells[idx] + input_gate * input_mod
            self.cells[idx + 1] = next_cell
            out = self.output.process(inp[idx], self.hidden[idx], self.sigmoid)
            next_hidden = out * self.tanh(next_cell)
            self.hidden[idx + 1] = next_hidden
            self.forget_data[idx] = forget_gate
            self.input_data[idx] = input_gate
            self.input_mod_data[idx] = input_mod
            self.output_data[idx] = out
        real_output = self.why @ next_hidden + self.by
        return real_output
    
    @staticmethod
    def softmax(xs):
        return np.exp(xs) / sum(np.exp(xs))
    
    def test(self, x: np.ndarray, return_argmax=True):
        """return softmax of the forward result\n
        x: 1d array or 2d array of samples\n
        return_argmax: return as a class or not"""
        if len(x.shape) == 1:
            x.reshape(x.shape + tuple(1))
        out = self.forward(x)
        out = self.softmax(out)
        if return_argmax:
            m = np.argmax(out)
            zeros = np.zeros(self.by.shape).tolist()
            zeros[m] = [1]
            return np.array(zeros)
        return out
    
    def backpropagation(self, X: np.ndarray, target: np.ndarray, learning_rate=0.01):
        target.flatten()
        probs = self.test(X, return_argmax=False)
        loss = -np.log(probs[np.where(target == 1)])
        probs[np.where(target == 1)] -= 1

        # backpropagation time
        # output layer
        last_layer_input = self.hidden[len(self.hidden) - 1]

        # dL/dWhy = dL/dy * dy/dWhy = loss * last_layer_input
        # dL/dby = dL/dy * dy/dby = loss * 1 = loss
        d_Why = probs @ last_layer_input.T
        d_by = probs

        # build loss for next hidden layer
        # dL/dh = dL/dy * dy/dh = loss * Why
        dL_dh = self.why.T @ probs

        # build dW and db for all gates
        doutput = dW_dB(self.output.xh.shape, self.output.hh.shape, self.output.bh.shape)
        dinputmod = dW_dB(self.inputmodulation.xh.shape, self.inputmodulation.hh.shape, self.inputmodulation.bh.shape)
        dinput = dW_dB(self.input.xh.shape, self.input.hh.shape, self.input.bh.shape)
        dForget = dW_dB(self.forget.xh.shape, self.forget.hh.shape, self.forget.bh.shape)
        
        # finding derivative stuff
        for i in reversed(range(len(self.hidden) - 1)):
            # cell
            dL_dc = dL_dh * self.output_data[i] * self.dtanh_dx(self.tanh(self.cells[i + 1]))

            # output
            temp = learning_rate * dL_dh * self.tanh(self.cells[i + 1]) * self.dsigmoid_dx(self.output_data[i])
            dL_dWx = temp @ self.last_input[i].T
            dL_dWh = temp @ self.hidden[i].T 
            dL_dbh = temp
            doutput += (dL_dWx, dL_dWh, dL_dbh)

            # input modulation
            temp = learning_rate * dL_dc * self.input_data[i] * self.dtanh_dx(self.input_mod_data[i])
            dL_dWx = temp @ self.last_input[i].T
            dL_dWh = temp @ self.hidden[i].T
            dL_dbh = temp
            dinputmod += (dL_dWx, dL_dWh, dL_dbh)

            # input
            temp = learning_rate * dL_dc * self.input_mod_data[i] * self.dsigmoid_dx(self.input_data[i])
            dL_dWx = temp @ self.last_input[i].T
            dL_dWh = temp @ self.hidden[i].T
            dL_dbh = temp
            dinput += (dL_dWx, dL_dWh, dL_dbh)

            # forget
            temp = learning_rate * dL_dc * self.cells[i] * self.dsigmoid_dx(self.forget_data[i])
            dL_dWx = temp @ self.last_input[i].T
            dL_dWh = temp @ self.hidden[i].T
            dL_dbh = temp
            dForget += (dL_dWx, dL_dWh, dL_dbh)

            # change dL/dc
            # dL/dc * dc/dc-1 = dL/dc-1
            dL_dc *= self.forget_data[i]
        
        # gradient descent with overload function
        self.forget -= dForget
        self.input -= dinput
        self.inputmodulation -= dinputmod
        self.output -= doutput
        self.why -= d_Why
        self.by -= d_by

        # return loss
        return loss