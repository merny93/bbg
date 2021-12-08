from bbg import BLG_H
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvalsh


N_k = 401
a0=0.246
Kmax = 0.025/a0*2
kxs = np.linspace(-Kmax,Kmax,N_k) + 2*np.pi/(3*a0)
kys = np.linspace(-Kmax,Kmax,N_k) + 2*np.pi/(3*np.sqrt(3)*a0)

H = BLG_H([kxs, kys], D=1.0)
evals = eigvalsh(H)

# points = np.pad(np.diff(np.sign(np.diff(evals[:,N_k//2,2]))) != 0, 1, constant_values = False)
# K_splits = kys[points]
# E_splits = evals[points, N_k//2, 2]
# ws = list(E_splits) + [(E_splits[0] + E_splits[1])/2, (E_splits[1] + E_splits[2])/2, 1.5*E_splits[1] + 0.5*E_splits[2]]
# ws.sort()

ws = np.linspace(0.045, 0.055, 15)
ws = list(ws) + [1,]
plt.contourf(kxs,kys, evals[:,:,2], levels = ws, vmax = 0.055, vmin = 0.045, cmap = "hot")
plt.colorbar()
plt.gca().set_aspect(1)
plt.xlabel("$k_x$ nm$^{-1}$")
plt.ylabel("$k_y$ nm$^{-1}$")
plt.savefig("surface.pdf")
