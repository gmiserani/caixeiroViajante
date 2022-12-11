'''
Nome: Gabriela Moraes Miserani de Freitas
Matricula: 2020006566
email: gmiserani@ufmg.br
'''


import numpy as np
import pandas as pd
import scipy as spdef 
import heapq as heap
import networkx as nx
import time #usado para fazer o teste de tempo

'''
função bound é usada na função do branch and bound para calcular a estimativa inicial do peso.
para isso, ela pega os valores das duas menores arestas associadas com cada vertice. como elas
são contadas duas vezes, dividimos a soma delas por dois e arredondamos para cima
'''
def bound(matrix, size):
    val = 0
    initialBound = []
    for i in range(size):
        lcopy = matrix[i].copy()
        mini = min(lcopy) #primeiro menor valor de aresta associado ao vertice i
        posi_mini = np.argmin(lcopy) #qual o vertice que esta ligado a i pela aresta minima
        lcopy[posi_mini] = np.inf #muda o valor para nao pegar o mesmo minimo duas vezes
        mini2 = min(lcopy) #segundo maior valor de aresta associado a i
        posi_mini2 = np.argmin(lcopy) #qual vertice está ligado a i pela segunda menor aresta
        vertice = i
        boundi = (posi_mini, posi_mini2) #faz uma tupla com os vertices ligados a i na estimativa atual
        tuple1 = (vertice, boundi) #associa o vertice i à tupla acima
        initialBound.append(tuple1) #insere essa associaçao na estimativa inicial
        val = val + mini + mini2
    val = val/2
    val = np.ceil(val)
    return val, initialBound #valor da estimativa e lista com as ligaçoes

'''
essa funçao atualiza a estimativa forçando a inserçao das arestas inclusas na soluçao que temos até agora
'''
def nextbound(dicionario1, CurrentSolution, next, matrix, size1):
    dicionario = dicionario1.copy()
    size = len(CurrentSolution) 
    if size == size1: #caso seja a volta para o primeiro vertice do ciclo, ou seja, o bound final
        ultimoVetice = CurrentSolution[-1]
        penultimovertice = CurrentSolution[-2]
        primeirovetice = CurrentSolution[0]
        segundovertice = CurrentSolution[1]
        tupladequemtaligado1 = list(dicionario[primeirovetice]) #tupla com que o primeiro vertice da solucao está ligado
        tupladequemtaligado = list(dicionario[ultimoVetice]) #tupla com que o ultimo vertice da solucao está ligado
        
        if tupladequemtaligado[0] != 0 and tupladequemtaligado[1] != 0: #ultimo vertice ja nao estava ligado ao primeiro
            if (tupladequemtaligado[0] == penultimovertice): #verifica qual já não está forçado ao penultimo vertice
                tupladequemtaligado[1] = 0 #liga ao primeiro
            else:
                tupladequemtaligado[0] = 0 #liga ao primeiro
        if (tupladequemtaligado1[0] == segundovertice): #faz a mesma coisa, mas com o primeiro vertice
            tupladequemtaligado1[1] = ultimoVetice
        else:
            tupladequemtaligado1[0] = ultimoVetice
        dicionario[ultimoVetice] = tuple(tupladequemtaligado) #atualiza os novos valores no dicionario
        dicionario[primeirovetice] = tuple(tupladequemtaligado1)
        
    else:
        if size == 1: #quando se está adicionando o segundo valor ao conjunto solução
            primeirovetice = CurrentSolution[0]
            primeiroVerticelig = list(dicionario[primeirovetice])
            segundovertice = next
            segundoverticelig = list(dicionario[segundovertice])
            #verifica as distacias entre o primeiro valor e os que está ligado pelo bound inicial
            distanciadoprimeiroverticeproprimeirobound = matrix[primeirovetice][primeiroVerticelig[0]]
            distanciadoprimeiroverticeprosegundobound = matrix[primeirovetice][primeiroVerticelig[1]]
            #verifica se, pelo bound inicial, o primeiro vertice já nao estava ligado ao vertice inserido
            if primeiroVerticelig[0] != segundovertice and primeiroVerticelig[1] != segundovertice:
                if distanciadoprimeiroverticeproprimeirobound > distanciadoprimeiroverticeprosegundobound:
                    primeiroVerticelig[0] = next #força a ligaçao no lugar da maior das arestas do bound inicial
                else:
                    primeiroVerticelig[1] = next
            #faz a mesma coisa com o vertice que esta sendo inserido
            distanciadosegundoverticeproprimeirobound = matrix[segundovertice][segundoverticelig[0]]
            distanciadosegundoverticeprosegundobound = matrix[segundovertice][segundoverticelig[1]]
            if segundoverticelig[0] != 0 and segundoverticelig[0] != 0:
                if distanciadosegundoverticeproprimeirobound > distanciadosegundoverticeprosegundobound:
                    segundoverticelig[0] = 0
                else:
                    segundoverticelig[1] = 0
            dicionario[primeirovetice] = tuple(primeiroVerticelig)
            dicionario[segundovertice] = tuple(segundoverticelig)
        else: #caso geral, quando adiciona qualquer outro valor a solucao atual
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
    val = 0 #calculamos o novo bound do mesmo jeito que na funcao anterior
    for i in range(size1):
        tupla = dicionario[i]
        val += matrix[i][tupla[0]]
        val += matrix[i][tupla[1]]
    val = val/2
    val = np.ceil(val)
    return val, dicionario
'''
essa funçao realiza o branch and bound 
'''
def branchandbound(matrix, size):
    bounds, initialBound = bound(matrix, size)
    initialBound = dict(initialBound)
    level = 0
    cost = 0
    currentSolution = [0] #começa do vertice 0
    root = (bounds, level, cost, currentSolution, initialBound) #inicializa a raiz do caminho
    root = list(root) #transforma em lista para poder alterar
    queue = [] 
    heap.heappush(queue, root) #insere a raiz na lista de prioridade
    sol = [] #soluçao final atual (folha)
    best = np.inf #inicia o melhor custo como infinito
    while len(queue) != 0:
        node = queue.pop()
        if node[1] > size-1: #adicionou todos os vertices
            if best > node[2]:
                best = node[2] #atualiza o best caso o custo do caminho atual seja o melhor ate agora
                sol = node[3] #a solucao final atual é o ceminho associado a esse node
        elif node[0] < best: #se a estimativa for pior que o melhor ate agora nao precisa computar
            if node[1] < size - 1: #nao tiver iserindo o ultimo vertice
                for k in range(size): #pega vertice a vertice
                    bound1, initialBound = nextbound(node[4], node[3], k, matrix, size)
                    #verificamos se o vertice k ja nao foi inserido antes,
                    #se nao estamos tentando conectar k a ele mesmo, 
                    #e se a estimativa com a insercao de k e melhor que best
                    if (k not in node[3]) and  matrix[node[3][-1]][k] != np.inf and bound1 < best:
                        level = node[1] + 1
                        cost = node[2]+ matrix[node[3][-1]][k]
                        currentSolution1 = node[3].copy()
                        currentSolution1.append(k)
                        b = []
                        b.extend([bound1, level, cost, currentSolution1, initialBound]) #novo node
                        heap.heappush(queue, b)
            else: # mesma coisa doa nterior, mas quando está fazendo a volta pro ponto inicial
                bound2, initialBound = nextbound(node[4], node[3], 0, matrix, size)
                if matrix[node[3][-1]][0] != np.inf and bound2 < best and len(node[3]) == size:
                    bound3, initialBound = nextbound(node[4], node[3], 0, matrix, size)
                    node[1] = node[1] + 1
                    node[2] = node[2] + matrix[node[3][-1]][0]
                    node[3].append(0)
                    a = []
                    a.extend([bound3, node[1], node[2], node[3], initialBound])
                    heap.heappush(queue, a) 
    return sol, best #depois de analizar todos os branchs validos e suas folhas, retornamos as solucao
'''
algoritmo twice around the tree
usa a biblioteca networkx para a manipulaçao de grafos
'''
def approxTSPtour(matrix, size):
    root = 0
    #root = np.random.random_integers(0, size-1) #possibilidade de se inicializar por qualquer vertice, e nao apenas pelo 0
    #criando um grafo a partir da matriz de pesos
    G = nx.complete_graph(size) 
    for (u,v) in G.edges():
        G.edges()[u,v]['weight'] = matrix[u][v]
    arvoreGeradoraMinima = nx.minimum_spanning_tree(G, algorithm='prim')
    CaminhamentoEuleriano = nx.dfs_preorder_nodes(arvoreGeradoraMinima, source=root)
    CaminhamentoEuleriano = list(CaminhamentoEuleriano)
    caminhoHamiltoniano = CaminhamentoEuleriano + [root] #retorno oa vertice inicial
    #calculo distancia(para uso nos testes)
    distancia = 0
    for i in range(size):
        distancia+= matrix[caminhoHamiltoniano[i]][caminhoHamiltoniano[i+1]]
    return caminhoHamiltoniano, distancia
'''
algoritmo de christofides
também usa a biblioteca networkx para manipulacao de grafos
'''
def christofides(matrix, size):
    #inicio igual ao do algoritmo twice around the tree
    root = 0
    #root = np.random.random_integers(0, size-1) 
    G = nx.complete_graph(size)
    for (u,v) in G.edges():
        G.edges()[u,v]['weight'] = matrix[u][v]
    arvoreGeradoraMinima = nx.minimum_spanning_tree(G, algorithm='prim')
    I = [] #armazena im I os vertices de grau impar
    for i in arvoreGeradoraMinima.nodes:
        if arvoreGeradoraMinima.degree[i] % 2 != 0:
            I.append(i)
    subGraph = G.subgraph(I) #encontra subgrafo de G contendo apenas os vertices de I
    matchingPerfeitoMinimo = nx.min_weight_matching(subGraph) #matching perfeito de peso minimo do subgrafo encontrado
    New = arvoreGeradoraMinima #New será a arvore geradora minima acrescida das arests do matching anterior
    for (u, v) in matchingPerfeitoMinimo:
        New.add_edge(u, v, weight=matrix[u][v])
    #final igual ao do algoritmo anterior
    CaminhamentoEuleriano = nx.dfs_preorder_nodes(New, source=root)
    CaminhamentoEuleriano = list(CaminhamentoEuleriano)
    caminhoHamiltoniano = CaminhamentoEuleriano + [root]
    #calculo distancia(para uso nos testes)
    distancia = 0
    for i in range(size):
        distancia+= matrix[caminhoHamiltoniano[i]][caminhoHamiltoniano[i+1]]
    return caminhoHamiltoniano, distancia

'''
funcao usada para gerar os pontos que serao usados nos algoritmos acima
'''
def genDots(size):
    #gera os valores aleatoriamente
    y = np.random.random_integers(0, 400, size=size)
    x = np.random.random_integers(0, 2000, size=size)
    x = np.unique(x) #verifica que os valores de x sejam unicos para que nao hajam pontos repetidos
    while len(x) < size:
        x = np.append(x, np.random.random_integers(0,2000))
        x = np.unique(x)
    return x,y

#calcula a distancia euclidiana entre os pontos e coloca na matrix
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

#calcula a distancia de manhattan entre os pontos e coloca na matrix
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

#inicializa a matriz usada nas funcoes acima
def matrixgen(x, y, size):
    matrix=[]
    for i in range(size):
        matrix.append([0 for i in range(size)])
    return matrix

'''
funcao que chama os algoritmos e imprime os resultados.
recebe um caractere que representa qual o calculo de distancia requerido:
m -> manhattan
e -> euclidiana
recebe um caractere que representa o algoritmo a ser usado:
t -> twice around the tree
c -> christofides
b -> branch and bound
'''
def chamadora(distancia, algoritmo, tamanho):
    x, y= genDots(tamanho)
    matrix = matrixgen(x, y, tamanho)
    if distancia == 'm':
        matrix = Manhattan(x, y, matrix, tamanho)
    elif distancia == 'e':
        matrix = Euclidiana(x, y, matrix, tamanho)
    if algoritmo == 't':
        print(approxTSPtour(matrix, tamanho))
    elif algoritmo == 'c':
        print(christofides(matrix, tamanho))
    elif algoritmo == 'b':
        print(branchandbound(matrix, tamanho))
    
#para a primeira instancia do problema
#chamadora('e', 'c', 2**4)