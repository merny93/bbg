from bbg import BLG_lin_K, DOS, DOS_fast,DOSk, BLG_H
import matplotlib.pyplot as plt
from time import perf_counter
import numpy as np
from numpy.linalg import eigvalsh
# from scipy.linalg import eigvalsh
N_k = 401
a0=0.246
Kmax = 0.1/a0*2
Kxs = np.linspace(-Kmax,Kmax,N_k) + 2*np.pi/(3*a0)
Kys = np.linspace(-Kmax,Kmax,N_k) + 2*np.pi/(3*np.sqrt(3)*a0)
Ds = np.arange(10)*0.1
num_Ds = len(Ds)

Params = dict({'g0':-2.61, 'g1':0.361 , 'g3':0.283, 'g4':0.138, 'Dp':0.015})

evals = np.zeros((num_Ds, len(Kxs), len(Kys), 4), dtype=float)
t1 = []
t2 =[]
for (k,D) in enumerate(Ds):
    tt1 = perf_counter()
    Hs = BLG_H([Kxs,Kys], Params, D)
    tt2 = perf_counter()
    evals[k,...] = eigvalsh(Hs)
    tt3 = perf_counter()
    t1.append(tt2-tt1)
    t2.append(tt3-tt2)
evals = np.transpose(evals,(0,3,1,2))
plt.imshow(evals[2, 2, :,:])
plt.show()
print(evals.shape)
print(sum(t1), sum(t2))
print(np.min(evals), np.max(evals))
ws = np.linspace(-1,1,1000)
doss = sum([DOS_fast(ws, evals[2, i,:,:]) for i in range(4)] )
plt.clf()
plt.plot(1000*ws, doss)
plt.show()
exit()
ws_slice_t = [0.0445, 0.0449, 0.0452, 0.048]
# ws_slice_t = [2, 3, 4,5]
fig,ax = plt.subplots(1,4,figsize=(20,5))
for i in range(4):
    css = [ax[i].contour(evals[-1,j,:], levels=[ws_slice_t[i]]) for j in range(4)]
    # ax[i].imshow(DOSk_t[i].T, origin='lower', cmap='Blues', extent=(-0.16,0.16,-0.16,0.16))
    ax[i].set_xlabel(r'$k_{x} [1/a_{0}]$')
    ax[i].set_title(f'$\omega=${ws_slice_t[i]*1000:.1f} meV')
ax[0].set_ylabel(r'$k_{y} [1/a_{0}]$')
fig.suptitle('Conduction band FS (electron-doping regime)')
plt.show()

# exit()

ws = np.linspace(0.044986,0.0450,100)
ws = np.linspace(0.04,0.1, 100)
plt.ion()
plt.clf()
for w in ws:
    plt.clf()
    css = [plt.contour(evals[-1,j,:,:], levels=[w]) for j in range(4)]
    plt.gca().set_aspect(1)
    plt.show()
    print(w)
    plt.pause(0.1)

# eta = 0.0001
# doss = np.array( [DOS(ws, evals[:,:,:,i], eta) for i in range(num_Ds)])
# plt.imshow(doss)
# plt.show()