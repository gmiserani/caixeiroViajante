import numpy as np
import pandas as pd
import scipy as spdef 
import networkx as nx

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