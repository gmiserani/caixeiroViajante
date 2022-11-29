import numpy as np
#import pandas as pd
import scipy as sp

def bound(matrix, size):
    val = 0
    initialBound = []
    for i in range(size):
        lcopy = matrix[i].copy()
        mini = min(lcopy)
        lcopy.remove(mini)
        mini2 = min(lcopy)
        tuple1 = (mini, mini2)
        initialBound.append(tuple1)
        val = val + mini + mini2
    val = val/2
    val = np.ceil(val)
    return val, initialBound

def nextbound(initialBound, CurrentSolution, next):
    initialCopy = initialBound.copy()
        #pegar o maior dos valores de cada tupla e substituir pelo da currentsolution e ai depois soma tudo
        #divide por dois e pega o teto
    size = np.size(initialBound)
    val = 0
    for i in range(size):
        val += i[0]
        val += i[1]
    val = val/2
    val = np.ceil(val)
    return val
    

def branchandbound(matrix, size):
    bound, initialBound = bound(matrix, size)
    level = 0
    cost = 0
    currentSolution = []
    root = (bound, level, cost, currentSolution)
    queue = []
    queue.append(root)
    best = np.inf
    sol = []
    while np.size(queue) != 0:
        node = queue.pop()
        if node[1] > size-1:
            if best > node[2]:
                best = node[2]
                sol = node[3]
        elif node[0] < best:
            if node[1] < size:
                for k in range(size-1):
                    if k in node[3] and matrix[node[3][-1]][k] != np.inf and nextbound(initialBound, node[3], k) < best:
                        queue.append((nextbound(initialBound, node[3], k), node[1] + 1, node[2]+matrix[node[3][-1]][k], node[3].append(k)))
            elif matrix[node[3][-1]][0] != np.inf and nextbound(initialBound, node[3], 0) < best and np.size(node[3]) == size:
                queue.append((nextbound(initialBound, node[3], 0), node[1]+1, node[2] + matrix[node[3][-1]][0], node[3].append(0))) 
