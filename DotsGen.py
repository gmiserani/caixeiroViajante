import numpy as np
import pandas as pd
import scipy as spdef 
import heapq as heap
import networkx as nx

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
    size = np.size(CurrentSolution)
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
    sol = []
    best = np.inf
    while np.size(queue) != 0:
        node = queue.pop()
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
                        heap.heappush(queue, b)
            else:
                bound2, initialBound = nextbound(node[4], node[3], 0, matrix, size)
                if matrix[node[3][-1]][0] != np.inf and bound2 < best and np.size(node[3]) == size:
                    bound3, initialBound = nextbound(node[4], node[3], 0, matrix, size)
                    node[1] = node[1] + 1
                    node[2] = node[2] + matrix[node[3][-1]][0]
                    node[3].append(0)
                    a = []
                    a.extend([bound3, node[1], node[2], node[3], initialBound])
                    heap.heappush(queue, a) 
    return sol

def approxTSPtour(matrix, size):
    root = 0
    #root = np.random.random_integers(0, size-1)
    G = nx.complete_graph(size)
    for (u,v) in G.edges():
        G.edges()[u,v]['weight'] = matrix[u][v]
    arvoreGeradoraMinima = nx.minimum_spanning_tree(G, algorithm='prim')
    CaminhamentoEuleriano = nx.dfs_preorder_nodes(arvoreGeradoraMinima, source=root)
    CaminhamentoEuleriano = list(CaminhamentoEuleriano)
    caminhoHamiltoniano = CaminhamentoEuleriano + [root]
    return caminhoHamiltoniano

def christofides(matrix, size):
    root = 0
    #root = np.random.random_integers(0, size-1)
    G = nx.complete_graph(size)
    for (u,v) in G.edges():
        G.edges()[u,v]['weight'] = matrix[u][v]
    arvoreGeradoraMinima = nx.minimum_spanning_tree(G, algorithm='prim')
    I = []
    for i in arvoreGeradoraMinima.nodes:
        if arvoreGeradoraMinima.degree[i] % 2 != 0:
            I.append(i)
    subGraph = G.subgraph(I)
    matchingPerfeitoMinimo = nx.min_weight_matching(subGraph)
    New = arvoreGeradoraMinima
    for (u, v) in matchingPerfeitoMinimo:
        New.add_edge(u, v, weight=matrix[u][v])
    CaminhamentoEuleriano = nx.dfs_preorder_nodes(New, source=root)
    CaminhamentoEuleriano = list(CaminhamentoEuleriano)
    caminhoHamiltoniano = CaminhamentoEuleriano + [root]
    return caminhoHamiltoniano

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
                dist = np.inf #mudar para zero
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


#matriz = [[np.inf, 3, 1, 5, 8], [3, np.inf, 6, 7, 9], [1, 6, np.inf, 4, 2], [5, 7, 4, np.inf, 3], [8, 9, 2, 3, np.inf]]

#matrizx = [[np.inf, 4, 8, 9, 12], [4, np.inf, 6, 8, 9], [8, 6, np.inf, 10, 11], [9, 8, 10, np.inf, 7], [12, 9, 11, 7, np.inf]]

#matrizc = [[np.inf, 4, 8, 9, 12], [4, np.inf, 6, 8, 9], [8, 6, np.inf, 10, 11], [9, 8, 10, np.inf, 7], [12, 9, 11, 7, np.inf]]


def chamadora(distancia, algoritmo, tamanho):
    x, y= genDots(tamanho)
    matrix = matrixgen(x, y, tamanho)
    if distancia == 'm':
        matrix = Manhattan(x, y, matrix, tamanho)
    elif distancia == 'e':
        matrix = Euclidiana(x, y, matrix, tamanho)
    if algoritmo == 'b':
        print(branchandbound(matrix, tamanho))
    elif algoritmo == 't':
        print(approxTSPtour(matrix, tamanho))
    elif algoritmo == 'c':
        print(christofides(matrix, tamanho))