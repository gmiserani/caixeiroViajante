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

def Euclidiana(x, y, matrix, size):
    for i in range(size):
        for j in range(size):
            if(i == j):
                dist = np.inf
            else:
                dist = np.sqrt(((x[i]-x[j])**2)+((y[i]-y[j])**2))
                dist = abs(dist)
            matrix[i][j] = dist
            matrix[j][i] = dist
    return matrix

def Manhattan(x, y, matrix, size):
    for i in range(size):
        for j in range(size):
            if(i == j):
                dist = np.inf
            else:
                dist = abs(x[i] - x[j]) + abs(y[i] - y[j])
            matrix[i][j] = dist
            matrix[j][i] = dist
    return matrix

def matrixgen(x, y, size):
    matrix=[]
    for i in range(size):
        matrix.append([0 for i in range(size)])
    return matrix

x, y= genDots(4)
matrix = matrixgen(x, y, 4)
print(Manhattan(x, y, matrix, 4))

#distancia euclidiana vs distancia de manhattan