from bbg import BLG_H, DOS_fast
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvalsh


N_k = 101
N_d = 51
N_w = 51
a0=0.246
Kmax = 0.04/a0*2
kxs = np.linspace(-Kmax,Kmax,N_k) + 2*np.pi/(3*a0)
kys = np.linspace(-Kmax,Kmax,N_k) + 2*np.pi/(3*np.sqrt(3)*a0)
Ds = np.linspace(0, 1, N_d)
Hs = np.array([BLG_H([kxs, kys], D=D) for D in Ds])
evals = eigvalsh(Hs)

ws = np.linspace(0,0.08, N_w)
# plt.contour(kxs,kys, evals[-1,:,:,2], levels=[ws[-1]])
# plt.show()    
doss = np.array([DOS_fast(ws, evals[i, :,:,2]) for i in range(N_d)])
plt.imshow(doss)
plt.show()
print(evals.shape)