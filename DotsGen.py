import numpy as np
#import pandas as pd
#import scipy as spdef 

def genDots(size):
    y = np.random.random_integers(0, 400, size=size)
    x = np.random.random_integers(0, 400, size=size)
    x = np.unique(x)
    while len(x) < size:
        x = np.append(x, np.random.random_integers(0,400))
        x = np.unique(x)
    return x,y
def matrixgen(x, y, size):
    matrix=[]
    for i in range(size):
        matrix.append([1 for i in range(size)])
    for j in range(size):
        matrix[j][j] = 0
    return matrix

x, y= genDots(4)
matrixgen(x, y, 4)