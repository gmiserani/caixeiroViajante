import numpy as np
import pandas as pd
import scipy as spdef 
import networkx as nx

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


