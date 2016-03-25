import numpy as np

def relu(a):
    a[a < 0] = 0

def tanh(a):
    np.tanh(a, a)

