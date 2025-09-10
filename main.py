import numpy as np
import polars as pd

data = pd.read_csv("./datasets/mnist_test.csv")

m, n = data.shape

data = np.array(data)
np.random.shuffle(data)

data_dev = data[:1000].T
Y_dev = data[0]

X_dev = data[1: n]

data_train = data[1000:m].T

Y_train = data_train[0]
X_train = data_train[1:n]


def init_params():
    W1 = np.random.rand(10, 784)
    b1 = np.random.rand(10, 1)

    W2 = np.random.rand(10, 784)
    b2 = np.random.rand(10, 1)

    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def SoftMax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))

def forward_propagation(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)

    Z2 = W2.dot(A1) + b2
    A2 = SoftMax(A1)

