import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy import sparse

N = 200
t = 1
k = np.pi*(np.sqrt(5)-1)


indices=np.zeros(2*N)
indptr=np.zeros(N+1)
indices[0]=1
indices[1]=N-1
indptr[0]=0
for i in range(N-2):
  indices[2*(i+1)]=i
  indices[2*i+3]=i+2
  indptr[(i+1)]=2*i+2
indices[2*N-2]=0
indices[2*N-1]=N-2
indptr[N-1]=2*(N-1)
indptr[N]=2*N
data=np.ones(2*N)
H=t*sparse.csr_matrix((data,indices,indptr)).todense()

def getdiag(Delta, phi):
    dia = np.zeros(N)
    for i in range(N):
        dia[i] = Delta*np.cos(k*i + phi)
    return dia

def getH(Delta, phi):
    dia = getdiag(Delta,  phi)
    for i in range(N):
        H[i,i] = dia[i]
    return H


def GS_wv(H):
    e, v = linalg.eigh(H)
    return v[:, np.argsort(e)[:N/2]]

# first N/2 entries are occupied, second N/2 entries are empty
# -> hop one particle means exchanging two entries
def propose_update(y):
    occ = np.random.randint(N/2)
    emp = np.random.randint(N/2, N)
    y[occ], y[emp] = y[emp], y[occ]

def update(y, D, wv, occ, emp):
    yt = list(y)
    yt[occ], yt[emp] = yt[emp], yt[occ]
    r = np.random.random()
    Dt = np.linalg.det(wv[yt[:N/2], :])
    p = min(1, (Dt/D)**2)
    if r < p:
        y = list(yt)
        D = Dt
    return y, D

def sweep(y, D, wv):
    order = range(N/2)
    np.random.shuffle(order)
    for occ in order:
        emp = np.random.randint(N/2, N)
        y, D = update(y, D, wv, occ, emp)

    return y, D

configs = []
labels = []
Neq = 50
Nconfig = 200
Nsweeps = 10

Deltas = np.linspace(0, 4, 20)
phi = 0#np.pi/4.
for i, Delta in enumerate(Deltas):
    print("configs for Delta = %.2f (%i/%i)"%(Delta, i+1, len(Deltas)))
    H = getH(Delta, phi)
    wv = GS_wv(H)
    y = range(N)
    D=0
    while D == 0:
        np.random.shuffle(y)
        D = np.linalg.det(wv[y[:N/2], :])
    for _ in range(Neq):
        y, D = sweep(y, D, wv)
    for ll in range(Nconfig*Nsweeps):
        y, D = sweep(y, D, wv)
        if ll%Nsweeps == 0:
            loc = np.zeros(N)
            loc[y[:N/2]] = 1
            configs.append(loc.copy())
            labels.append(Delta)

np.savetxt("train_configs_L_%i.txt" %N, configs, fmt='%.2e')
np.savetxt("train_labels_L_%i.txt" %N, labels, fmt='%.2e')
