from bbg import BLG_H
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvals as eigvalsh
from scipy import ndimage

N_k = 401
a0=0.246
Kmax = 0.025/a0*2
kxs = np.linspace(-Kmax,Kmax,N_k) + 2*np.pi/(3*a0)
kys = np.linspace(-Kmax,Kmax,N_k) + 2*np.pi/(3*np.sqrt(3)*a0)

H = BLG_H([kxs, kys], D=1.0)
evals = eigvalsh(H)
print(np.mean(np.abs(np.imag(evals))))
evals= np.real_if_close(evals)
evals = np.sort(evals, axis =-1)
original = evals[:,:,2]
plt.imshow(original)
plt.show()
rotated = ndimage.rotate(original,120, reshape=False)
rotated2 =ndimage.rotate(original,-120, reshape=False)
ws = np.linspace(0.045, 0.055, 15)
ws = list(ws) + [1,]
plt.contour(kxs,kys, original, levels=[0.0457], colors="red")
plt.contour(kxs,kys, rotated, levels=[0.0457], colors="red")
plt.contour(kxs,kys, rotated2, levels=[0.0457], colors="red")


##This line plots the original map with the countours overplotted
plt.imshow(np.where(np.logical_and(rotated!=0, rotated2!=0), original, np.nan),vmax= 0.0457, origin="lower", extent=(kxs[0], kxs[-1], kys[0], kys[-1]))

##This line plots an error representation of the rotated masks
# plt.imshow(np.where(np.logical_and(rotated!=0, rotated2!=0), np.abs(original-0.5*(rotated + rotated2)), np.nan), vmax = 4e-5, origin="lower", extent=(kxs[0], kxs[-1], kys[0], kys[-1]))


plt.colorbar()
plt.gca().set_aspect(1)
plt.xlabel("$k_x$ nm$^{-1}$")
plt.ylabel("$k_y$ nm$^{-1}$")
plt.show()
