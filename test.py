import numpy as np
import pandas as pd
import scipy as spdef 
import networkx as nx

matriz = [[np.inf, 3, 1, 5, 8], [3, np.inf, 6, 7, 9], [1, 6, np.inf, 4, 2], [5, 7, 4, np.inf, 3], [8, 9, 2, 3, np.inf]]
G = nx.complete_graph(5)
for (u,v) in G.edges():
    G.edges()[u,v]['weight'] = matriz[u][v]
print(G)
