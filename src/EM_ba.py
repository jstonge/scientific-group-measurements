"""
This notebook demonstrates the EM algorithm for the noise model.

    Author: Borrowed from Jean-Gabriel Young jean-gabriel.young@uvm.edu
    Semester: Spring 2023
    License: CC BY 4.0
"""

import numpy as np
import seaborn as sns
import itertools as it
sns.set_style('ticks')

def em_algo(E, N, n, theta_0, T):
    """Iterate the EM steps untill convergence.
    
    Parameters
    ----------
    E: data matrix
    N: number of samples
    n: number of nodes
    theta_0: initial parameters (alpha, beta, rho), as a tuple
    T: number of iterations
    """
    def e_step(alpha, beta, rho, E, N, n):
        """Compute the probability distribution q(A), i.e., the matrix Q."""
        # note: n, the number of node is passed for clarity. it can be derived from E too.
        Q = np.zeros((n, n))
        for i, j in it.combinations(range(n), 2):
            h = rho * alpha ** E[i, j] * (1 - alpha) ** (N - E[i,j])
            g = (1 - rho) * beta ** E[i, j] * (1 - beta) ** (N - E[i,j])
            Q[i, j] = h / (h + g)
            Q[j, i] = Q[i, j]
        return Q

    def m_step(Q, E, N, n):
        """Compute the best values of the parameters given Q."""
        # note: n, the number of node is passed for clarity. it can be derived from E too.
        alpha = np.sum(np.triu(E * Q, 1)) / (N *  np.sum(np.triu(Q, 1)))
        beta = np.sum(np.triu(E * (1 - Q), 1)) / (N *  np.sum(np.triu(1 - Q, 1)))
        rho = 2 * np.sum(np.triu(Q, 1)) / (n * (n - 1))
        return alpha, beta, rho

    # note: n, the number of node is passed for clarity. it can be derived from E too.
    
    # initialize
    alpha, beta, rho = theta_0
    trajectory = [] # store intermediary states
    
    # loop:
    for t in range(T):
        Q = e_step(alpha, beta, rho, E, N, n)
        (alpha, beta, rho) = m_step(Q, E, N, n)
        trajectory.append({"Q": Q, "alpha": alpha, "beta": beta, "rho":  rho})
    
    return trajectory