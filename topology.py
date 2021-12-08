from bbg import BLG_H
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvalsh


N_k = 801
a0=0.246
Kmax = 0.04/a0*2
kxs = np.linspace(-Kmax,Kmax,N_k) + 2*np.pi/(3*a0)
kys = np.linspace(-Kmax,Kmax,N_k) + 2*np.pi/(3*np.sqrt(3)*a0)

H = BLG_H([kxs, kys], D=1.0)
evals = eigvalsh(H)
print(evals.shape)


points = np.pad(np.diff(np.sign(np.diff(evals[:,N_k//2,2]))) != 0, 1, constant_values = False)
K_splits = kys[points]
E_splits = evals[points, N_k//2, 2]
ws = list(E_splits) + [(E_splits[0] + E_splits[1])/2, (E_splits[1] + E_splits[2])/2, 1.5*E_splits[1] + 0.5*E_splits[2]]
ws.sort()

# plt.imshow(np.ma.masked_where(evals[:,:,2] > ws[0]+0.0001, evals[:,:,2]))
# plt.show()


# ws = np.array(ws) + 0.0001
fig,ax = plt.subplots(2,3,figsize=(9,7))
ax_view = ax.reshape(-1)
epsilon = 0.000001
for i,w in enumerate(ws):
    levels = [w+(i*epsilon) for i in (-1,0,1)]
    ax_view[i].contour(kxs, kys,evals[:,:,2], levels=levels, colors = "blue")
    # ax_view[i].scatter([kxs[N_k//2]],[kys[N_k//2]])
    ax_view[i].set_xlabel(r'$k_{x} [1/a_{0}]$')
    ax_view[i].set_title(f'$\omega=${w*1000:.1f} meV')
    ax_view[i].set_aspect(1)
ax_view[0].set_ylabel(r'$k_{y} [1/a_{0}]$')
# fig.suptitle('Conduction band FS (electron-doping regime)')
plt.savefig("topology.pdf")
