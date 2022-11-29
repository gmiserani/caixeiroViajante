import numpy as np
import pandas as pd
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

#def nextBound(CurrentBound, next):

def branchandbound(matrix, size):
    val, initialBound = bound(matrix, size)
    level = 0
    cost = 0
    currentSolution = []
    root = (val, level, cost, currentSolution)
    queue = []
    queue.append(root)
    best = np.inf
    sol = []
    while np.size(queue) != 0:
        node = queue.pop()