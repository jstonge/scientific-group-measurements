import numpy as np
import networkx as nx
import itertools as it

def generate_error_model(g, alpha, beta):
    n = g.number_of_nodes()

    sample = nx.Graph()
    for i, j in it.combinations(range(n), 2):
        if g.has_edge(i, j):
            if np.random.rand() < alpha:
                sample.add_edge(i, j, etype='tp')
        else:
            if np.random.rand() < beta:
                sample.add_edge(i, j, etype='fp')
    return sample


def noisy_er(n=100, m=1, alpha=0.9, beta=0.001, N=3):
    g = nx.generators.barabasi_albert_graph(n, m)
    X = [generate_error_model(g, alpha, beta) for t in range(N)]
    return g, X