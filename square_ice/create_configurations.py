#-------------------------------------------------------------------------------
# Filename: create_configurations.py
# Description: creates monte carlo configurations for the ising square ice
# theory. 
# Note that this only creates zero and infinite temperature states.
# Also note that at the moment, I only construct zero-temperature states which
# have explicitly zero total magnetization. This is probably not very smart.
# Authors: Mark H Fischer
#-------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

N=16        #size of the system is NxN
J=1.        #only parameter of the Hamiltonian

N_low = 20000   #number of configurations at low (=0) temperature
N_high= 20000   #number of configurations at high (=np.inf) temperature
Nupdate = N

def initialize(ground_state=False):
    '''
    Initializes a random spin configuration on a square lattice
    i,j denotes the plaquette to the right / up from vertex i,j and xy usually denotes
        the spin at + x/2 or + y/2 to the center.

    Returns 
    -------
    Random spin configuration with format NxNx2 where
    '''
    if ground_state:
        spins = np.ones((N,N,2))
        spins[:,:,0]*=-1
    else:
        spins = 2*np.random.randint(2, size=((N, N,2))) - np.ones((N,N,2))
    return spins

def next_point(i,j,direction):
    if direction==0:
        return (i+1)%N, j
    if direction==1:
        return i, (j+1)%N
    if direction==2:
        return (i+N-1)%N, j
    if direction==3:
        return i, (j+N-1)%N
    return i,j

def next_spin_index(i,j,direction):
    if direction==0:
        return i,j,0
    if direction==1:
        return i,j,1
    if direction==2:
        return (i+N-1)%N, j,0
    if direction==3:
        return i, (j+N-1)%N,1
    return 1

def loop_update(spins):
    '''
    first creates a loop of spins to be flipped:
    at each vertex, choose the oposite spin for the outgoing leg compared to
    the incoming leg.
    Since this update does not change the energy, it's performed with probability 1.
    '''

    # pick a vertex
    spins_to_update = []
    loop = []
    i,j = np.random.randint(N, size=2)
    loop.append([i,j])
    direction = np.random.randint(4)
    i_old, j_old = next_point(i,j,direction)
    i_new = i_old
    j_new = j_old
    loop.append([i_new, j_new])
    first_spin = spins[next_spin_index(i,j,direction)]
    old_spin = first_spin
    spins_to_update.append(next_spin_index(i,j,direction))
    while not (i_new==i and j_new==j and not old_spin==first_spin):
        next_step = np.random.randint(4)
        i_new, j_new = next_point(i_old, j_old, next_step)
        new_index = next_spin_index(i_old, j_old, next_step)
        if not spins[new_index]==old_spin and not new_index in spins_to_update: 
            loop.append([i_new, j_new])
            spins_to_update.append(new_index)
            old_spin = spins[new_index]
            i_old = i_new
            j_old = j_new
        else:
            i_new = i_old
            j_new = j_old

    for spin_index in spins_to_update:
        spins[spin_index]*=-1

configs = []
labels = []
    

# First create some zero-temperature configurations
spins = initialize(ground_state=True)

for i in range(N_low*Nupdate):
    loop_update(spins)
    if i%Nupdate==0:
        configs.append(np.reshape(spins.copy(), N*N*2))
        labels.append(0)


# now for infinite temperature
spins = initialize()

for i in range(N_high):
    configs.append(np.reshape(initialize(), N*N*2))
    labels.append(1)

np.savetxt("configs.txt", configs, fmt='%i')
np.savetxt("labels.txt", labels, fmt='%i')

