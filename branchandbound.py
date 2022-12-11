import numpy as np
import heapq as heap
#import pandas as pd
import scipy as sp

def bound(matrix, size):
    val = 0
    initialBound = []
    for i in range(size):
        lcopy = matrix[i].copy()
        mini = min(lcopy)
        posi_mini = np.argmin(lcopy)
        lcopy[posi_mini] = np.inf #muda o valor para nao pegar o mesmo minimo duas vezes
        mini2 = min(lcopy)
        posi_mini2 = np.argmin(lcopy)
        vertice = i
        boundi = (posi_mini, posi_mini2)
        tuple1 = (vertice, boundi)
        initialBound.append(tuple1)
        val = val + mini + mini2
    val = val/2
    val = np.ceil(val)
    return val, initialBound

def nextbound(dicionario1, CurrentSolution, next, matrix, size1):
    dicionario = dicionario1.copy()
    size = len(CurrentSolution)
    if size == size1: #caso seja a volta para o primeiro vertice do ciclo
        ultimoVetice = CurrentSolution[-1]
        penultimovertice = CurrentSolution[-2]
        primeirovetice = CurrentSolution[0]
        segundovertice = CurrentSolution[1]
        tupladequemtaligado1 = list(dicionario[primeirovetice])
        tupladequemtaligado = list(dicionario[ultimoVetice])
        
        if tupladequemtaligado[0] != 0 and tupladequemtaligado[1] != 0:
            if (tupladequemtaligado[0] == penultimovertice):
                tupladequemtaligado[1] = 0
            else:
                tupladequemtaligado[0] = 0
        if (tupladequemtaligado1[0] == segundovertice):
            tupladequemtaligado1[1] = ultimoVetice
        else:
            tupladequemtaligado1[0] = ultimoVetice
        dicionario[ultimoVetice] = tuple(tupladequemtaligado)
        dicionario[primeirovetice] = tuple(tupladequemtaligado1)
        
    else:
        if size == 1:
            primeirovetice = CurrentSolution[0]
            primeiroVerticelig = list(dicionario[primeirovetice])
            segundovertice = next
            segundoverticelig = list(dicionario[segundovertice])
            distanciadoprimeiroverticeproprimeirobound = matrix[primeirovetice][primeiroVerticelig[0]]
            distanciadoprimeiroverticeprosegundobound = matrix[primeirovetice][primeiroVerticelig[1]]
            if primeiroVerticelig[0] != segundovertice and primeiroVerticelig[1] != segundovertice:
                if distanciadoprimeiroverticeproprimeirobound > distanciadoprimeiroverticeprosegundobound:
                    primeiroVerticelig[0] = next
                else:
                    primeiroVerticelig[1] = next
            distanciadosegundoverticeproprimeirobound = matrix[segundovertice][segundoverticelig[0]]
            distanciadosegundoverticeprosegundobound = matrix[segundovertice][segundoverticelig[1]]
            if segundoverticelig[0] != 0 and segundoverticelig[0] != 0:
                if distanciadosegundoverticeproprimeirobound > distanciadosegundoverticeprosegundobound:
                    segundoverticelig[0] = 0
                else:
                    segundoverticelig[1] = 0
            dicionario[primeirovetice] = tuple(primeiroVerticelig)
            dicionario[segundovertice] = tuple(segundoverticelig)
        else:
            ultimoateagora = CurrentSolution[-1]
            penultimoateagora = CurrentSolution[-2]
            tupladequemtaligado = list(dicionario[ultimoateagora])
            if tupladequemtaligado[0] != next and tupladequemtaligado[1] != next:
                if tupladequemtaligado[0] == penultimoateagora:
                    tupladequemtaligado[1] = next
                else:
                    tupladequemtaligado[0] = next
            dicionario[ultimoateagora] = tuple(tupladequemtaligado)
            nextlig = list(dicionario[next])
            distanciadonextproprimeirobound = matrix[next][nextlig[0]]
            distanciadonextprosegundobound = matrix[next][nextlig[1]]
            if nextlig[0] != ultimoateagora and nextlig[1] != ultimoateagora:
                if distanciadonextproprimeirobound > distanciadonextprosegundobound:
                    nextlig[0] = ultimoateagora
                else:
                    nextlig[1] = ultimoateagora
            dicionario[next] = tuple(nextlig)
        #pegar o maior dos valores de cada tupla e substituir pelo da currentsolution e ai depois soma tudo
        #divide por dois e pega o teto
    val = 0
    for i in range(size1):
        tupla = dicionario[i]
        val += matrix[i][tupla[0]]
        val += matrix[i][tupla[1]]
    val = val/2
    val = np.ceil(val)
    return val, dicionario
    
def branchandbound(matrix, size):
    bounds, initialBound = bound(matrix, size)
    initialBound = dict(initialBound)
    level = 0
    cost = 0
    currentSolution = [0]
    root = (bounds, level, cost, currentSolution, initialBound)
    root = list(root)
    queue = []
    heap.heappush(queue, root)
    print(queue)
    sol = []
    best = np.inf
    while len(queue) != 0:
        node = queue.pop()
        #print(node)
        #print(node[2])
        if node[1] > size-1:
            if best > node[2]:
                best = node[2]
                sol = node[3]
        elif node[0] < best:
            if node[1] < size - 1:
                for k in range(size):
                    bound1, initialBound = nextbound(node[4], node[3], k, matrix, size)
                    if (k not in node[3]) and  matrix[node[3][-1]][k] != np.inf and bound1 < best:
                        level = node[1] + 1
                        cost = node[2]+ matrix[node[3][-1]][k]
                        currentSolution1 = node[3].copy()
                        currentSolution1.append(k)
                        b = []
                        b.extend([bound1, level, cost, currentSolution1, initialBound])
                        print(b)
                        print(bound1)
                        heap.heappush(queue, b)
                        print("wtf")
            else:
                bound2, initialBound = nextbound(node[4], node[3], 0, matrix, size)
                if matrix[node[3][-1]][0] != np.inf and bound2 < best and len(node[3]) == size:
                    bound3, initialBound = nextbound(node[4], node[3], 0, matrix, size)
                    node[1] = node[1] + 1
                    node[2] = node[2] + matrix[node[3][-1]][0]
                    node[3].append(0)
                    a = []
                    a.extend([bound3, node[1], node[2], node[3], initialBound])
                    heap.heappush(queue, a) 
    print(best)
    return sol
