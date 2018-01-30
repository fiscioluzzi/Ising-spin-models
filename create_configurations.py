import numpy as np
import matplotlib.pyplot as plt

N=16
J=1.

def initialize():
    '''
    returns:
        random spin configuration with format NxNx2 where

    note:
        i,j denotes the plaquette to the right / up from vertex i,j and xy usually denotes
        the spin at + x/2 or + y/2 to the center.
    '''

    spins = 2*np.random.randint(2, size=((N, N,2))) - np.ones((N,N,2))
    return spins

def total_energy(spins):
    '''
    returns:
        total energy of the spin configuration

    input:
        spins     :  spin configuration
    '''

    N = np.shape(spins)[0]
    energy = 0
    for i in range(N):
        i_left = (i+N-1)%N
        for j in range(N):
            j_up = (N+j-1)%N
            en = -J*(spins[i,j, 0]*spins[i_left, j,
                0]*spins[i,j,1]*spins[i,j_up, 1])
            energy += en
    return energy

def dE(spins, i, j, xy):
    '''
    returns:
        energy difference after flipping the 'xy' spin at plaquette ('i','j')

    note:
        energy before flipping:
          for x spin:
            s(i,j)_x * [s(i-1,j)_x * s(i,j-1)_y * s(i,j)_y + s(i+1,j)_x * s(i+1,j-1)_y * s(i+1,j)_y]
          for y spin:
            s(i,j)_y * [s(i-1,j)_x * s(i,j-1)_y * s(i,j)_x + s(i-1,j+1)_x * s(i,j+1)_y * s(i,j+1)_x]

        after flipping, energy is -s(i,j)*[...] -> difference is 2*s(i,j)*[...]
    '''

    i_right = (i+1)%N
    i_left = (i+N-1)%N
    j_down = (j+1)%N
    j_up = (N+j-1)%N
    if xy == 0: # the x spin should be updated
        left_plaquette = spins[i,j,0]*spins[i_left, j,0]*spins[i,
                j_up,1]*spins[i,j,1]
        right_plaquette = spins[i,j,0]*spins[i_right, j,0]*spins[i_right,
                j_up,1]*spins[i_right,j,1]

        return 2*J*(left_plaquette+right_plaquette)
    else:
        up_plaquette = spins[i,j,1]*spins[i_left, j,0]*spins[i,
                j_up,1]*spins[i,j,0]
        down_plaquette = spins[i,j,1]*spins[i_left, j_down,0]*spins[i,
                j_down,1]*spins[i,j_down,0]

        return 2*J*(up_plaquette + down_plaquette)

def single_spin_update(spins, T):
    '''
    performs a single step in a Metropolis single-spin update

    input:
        spins  :  spin configuration
        T      :  temperature for the probability
    '''

    # first, choose the plaquette
    i,j = np.random.randint(N, size=2)
    # now choose whether to look at the x or y spin
    xy = np.random.randint(2)
    DE = dE(spins, i,j, xy)
    r = np.random.random()
    if T==0 and DE <= 0:
        spins[i,j,xy] *= -1
        return 
    if T==0: return
    if r < np.exp(-DE/T):
        spins[i,j,xy] *= -1

def vertex_update(spins):
    '''
    performs a vertex update, i.e., flipps all the spins around the vertex (i,j)
    Since this update does not change the energy, it's performed with probability 1.
    '''

    # pick a vertex
    i,j = np.random.randint(N, size=2)
    i_left = (i+N-1)%N
    j_up = (N+j-1)%N
    # and flip every spin connected to it
    spins[i_left, j, 0]*=-1
    spins[i_left, j_up, 0]*=-1
    spins[i, j_up, 1]*=-1
    spins[i_left, j_up, 1]*=-1


N_low = 2 #number of configurations at low (=0) temperature
N_high= 2 #number of configurations at high (=np.inf) temperature
Neq = 100000
Nupdate = N**2

configs = []
labels = []
    
spins = initialize()

# First create some zero-temperature configurations
not_yet = True
i=0
while not_yet:
    single_spin_update(spins, 0)
    vertex_update(spins)
    if i%100==0:
        if total_energy(spins)==-N**2: not_yet = False
    i+=1

for i in range(N_low*Nupdate):
    vertex_update(spins)
    if i%Nupdate==0:
        configs.append(np.reshape(spins.copy(), N*N*2))
        labels.append(0)


# now for infinite temperature
spins = initialize()
for _ in range(Neq):
    single_spin_update(spins, np.inf)
    vertex_update(spins)

for i in range(N_high*Nupdate):
    single_spin_update(spins, np.inf)
    vertex_update(spins)
    if i%Nupdate==0:
        configs.append(np.reshape(spins.copy(), N*N*2))
        labels.append(1)

np.savetxt("configs.txt", configs, fmt='%.2e')
np.savetxt("labels.txt", labels, fmt='%.2e')

