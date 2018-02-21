#-------------------------------------------------------------------------------
# Filename: create_configurations.py
# Description: creates monte carlo configurations for the ising model. The
# output is then saved in 'labels_NxN.txt' and 'configs_NxN.txt'. Note that for
# each temperature in 'labels', there is a line of length N*N in configs.
# theory. 
# Authors: Mark H Fischer
#-------------------------------------------------------------------------------

import numpy as np
from collections import deque

# size of the system NxN
N = 30
# exchange coupling
J=1
# uncomment to seed random number generator
# np.random.seed(10)

# number of cluster updates to thermalize
T_therm = 1000

def initialize():
    '''
    Initializes a random spin configuration on a square lattice

    Returns 
    -------
    Random spin configuration with format NxNx2 where
    '''

    return 2*np.random.randint(2, size=((N, N))) - np.ones((N,N))

def cluster_update(configuration, T):
    '''
    Performs a cluster update following the Wolff algorithm.

    Parameters
    ----------
    spins  :  int
        spin configuration, dimension is NxNx2
    T      :  double
        temperature for the probability

    Returns
    ---------
    size of cluster build and flipped.
    '''

    size = 0
    visited = np.zeros((N,N))
    cluster=[]
    # choose random initial spin
    i,j = np.random.randint(N, size=2)
    cluster.append((i,j))
    visited[i,j]=1
    while len(cluster)>0:
        i,j = cluster.pop() #next i,j in line
        i_left = (i+1)%N
        i_right = (i+N-1)%N
        j_up = (j+1)%N
        j_down = (N+j-1)%N
        neighbors = [(i_left, j), (i_right, j), (i, j_up), (i, j_down)]
        for neighbor in neighbors:
            if visited[neighbor]==0 and configuration[neighbor] == configuration[i,j] and np.random.random()< (1-np.exp(-2*J/T)):
                cluster.append(neighbor)
                visited[neighbor]=1
                size += 1
        configuration[i,j]*=-1
    return size


train_configs = []
train_labels = []

# how many temperaturs
num_T = 100
min_T = 1.0
max_T = 3.5

# how many configurations per temperature
num_conf = 100

# For no obvious reason, I pick 100 random temperatures between min_T and
# max_T.
Temps = min_T + np.random.random(num_T)*(max_T - min_T)
for i, T in enumerate(Temps):
    print("create configurations for T=%.4f (%i / %i)" %(T, i+1, len(Temps)))
    configuration = initialize()
    csize = []
    # This is really an ad-hoc solution to the 'uncorrelated configurations'
    # problem, i.e., during some thermalization, I calculate average cluster
    # size, then update roughly enough according to this size.
    for _ in range(T_therm):
        csize.append(cluster_update(configuration, T))
    T_A = int(N**2 / (2*np.mean(csize))) * 2 + 1
    for i in range(num_conf*T_A):
        cluster_update(configuration, T)
        if i%T_A == 0:
            train_configs.append(np.reshape(configuration.copy(), N**2))
            train_labels.append(T)

np.savetxt("labels_%ix%i.txt"%(N,N), train_labels, fmt='%.2e')
np.savetxt("configs_%ix%i.txt"%(N,N), train_configs,  fmt='%.2e')
