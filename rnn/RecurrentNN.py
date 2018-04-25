import numpy as np
from LSTM import *

class RecurrentNeuralNetwork:
    def __init__(self, x_size, y_size, rl, expOut, learnRate):
        self.x = np.zeros(x_size)
        self.xs = x_size
        self.y = np.zeros(y_size)
        self.ys = y_size
        self.w = np.random.random((y_size, y_size))
        self.G = np.zeros_like(self.w)
        self.rl = rl
        self.lr = learnRate
        self.ia = np.zeros((rl + 1, x_size))
        self.ca = np.zeros((rl + 1, y_size))
        self.oa = np.zeros((rl + 1, y_size))
        self.ha = np.zeros((rl + 1, y_size))
        self.af = np.zeros((rl + 1, y_size))
        self.ai = np.zeros((rl + 1, y_size))
        self.ac = np.zeros((rl + 1, y_size))
        self.ao = np.zeros((rl + 1, y_size))
        self.eo = np.vstack((np.zeros(expOut.shape[0]), expOut.T))
        self.LSTM = LSTM(x_size, y_size, rl, learnRate)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def forwardProp(self):
        for i in range(1, self.rl + 1):
            self.LSTM.x = np.hstack((self.ha[i - 1], self.x))
            cs, hs, f, inp, c, o = self.LSTM.forwardProp()
            self.ca[i] = cs
            self.ha[i] = hs
            self.af[i] = f
            self.ai[i] = inp
            self.ac[i] = c
            self.ao[i] = o
            self.oa[i] = self.sigmoid(np.dot(self.w, hs))
            self.x = self.eo[i - 1]
        return self.oa

    def backProp(self):
        totalError = 0
        dfcs = np.zeros(self.ys)
        dfhs = np.zeros(self.ys)
        tu = np.zeros((self.ys, self.ys))
        tfu = np.zeros((self.ys, self.xs + self.ys))
        tiu = np.zeros((self.ys, self.xs + self.ys))
        tcu = np.zeros((self.ys, self.xs + self.ys))
        tou = np.zeros((self.ys, self.xs + self.ys))
        for i in range(self.rl, -1, -1):
            error = self.oa[i] - self.eo[i]
            tu += np.dot(np.atleast_2d(error * self.dsigmoid(self.oa[i])), np.atleast_2d(self.ha[i]).T)
            error = np.dot(error, self.w)
            self.LSTM.x = np.hstack((self.ha[i - 1], self.ia[i]))
            self.LSTM.cs = self.ca[i]
            fu, iu, cu, ou, dfcs, dfhs = self.LSTM.backProp(error, self.ca[i - 1],
                                                            self.af[i], self.ai[i], self.ac[i],self.ao[i], dfcs, dfhs)
            totalError += np.sum(error)
            tfu += fu
            tiu += iu
            tcu += cu
            tou += ou
        self.LSTM.update(tfu / self.rl, tiu / self.rl, tcu / self.rl, tou / self.rl)
        self.update(tu / self.rl)
        return totalError

    def update(self, u):
        self.G = 0.9 * self.G + 0.1 * u ** 2
        self.w -= self.lr / np.sqrt(self.G + 1e-8) * u
        return

    def sample(self):
        for i in range(1, self.rl + 1):
            self.LSTM.x = np.hstack((self.ha[i - 1], self.x))
            cs, hs, f, inp, c, o = self.LSTM.forwardProp()
            maxI = np.argmax(self.x)
            self.x = np.zeros_like(self.x)
            self.x[maxI] = 1
            self.ia[i] = self.x  # Use np.argmax?
            self.ca[i] = cs
            self.ha[i] = hs
            self.af[i] = f
            self.ai[i] = inp
            self.ac[i] = c
            self.ao[i] = o
            self.oa[i] = self.sigmoid(np.dot(self.w, hs))
            maxI = np.argmax(self.oa[i])
            newX = np.zeros_like(self.x)
            newX[maxI] = 1
            self.x = newX
        return self.oa
