from bbg import BLG_H, DOS, DOS_fast
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvalsh
from multiprocessing import Pool
from itertools import cycle


if __name__ == "__main__":
    N_k = 201
    N_d = 151
    N_w = 151
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

    job = zip(cycle((ws,)), (evals[i, :,:,2] for i in range(N_d)))
    with Pool(4) as pool:
        res = pool.starmap(DOS_fast ,job)
    doss = np.array(res)
    # doss = np.array([DOS_fast(ws, evals[i, :,:,2]) for i in range(N_d)])
    plt.imshow(doss)
    plt.show()
    print(evals.shape)