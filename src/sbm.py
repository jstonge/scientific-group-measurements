from numpy import ones, array
from random import randrange
from scipy.stats import poisson
from networkx import Graph

def fast_sbm_k_groups(c_in, c_out, n, K):
    """
    Generate a stochastic block model graph with K groups.
    
    Parameters:
    c_in: Average number of in-group connections per node.
    c_out: Average number of out-group connections per node.
    n: Total number of nodes.
    K: Number of communities.
    """
    # Calculate average connections based on the community size
    community_size = n // K
    mav_in = c_in * community_size / (K - 1)
    mav_out = c_out * community_size / (K - 1)
    
    G = Graph()
    for i in range(n):
        G.add_node(i)

    # Assign nodes to communities
    communities = {i: range(i * community_size, (i + 1) * community_size) for i in range(K)}
    
    # Generate in-group edges
    for comm in communities.values():
        m_in = poisson.rvs(mav_in)
        counter = 0
        while counter < m_in:
            u = randrange(comm.start, comm.stop)
            v = randrange(comm.start, comm.stop)
            if u != v:
                G.add_edge(u, v)
                counter += 1
                
    # Generate out-group edges
    for i in range(K):
        for j in range(i + 1, K):
            m_out = poisson.rvs(mav_out)
            counter = 0
            while counter < m_out:
                u = randrange(communities[i].start, communities[i].stop)
                v = randrange(communities[j].start, communities[j].stop)
                if u != v:
                    G.add_edge(u, v)
                    counter += 1

    return G
