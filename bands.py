from bbg import BLG_H
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvalsh

N_k = 401
a0=0.246
Kmax = 0.03/a0*2
kxs = np.linspace(-Kmax,Kmax,N_k) + 2*np.pi/(3*a0)
kxs = np.array([0])+ 2*np.pi/(3*a0)
kys = np.linspace(-Kmax,Kmax,N_k) + 2*np.pi/(3*np.sqrt(3)*a0)

H = BLG_H([kxs, kys], D=1.0)
evals = eigvalsh(H)
print(evals.shape)

# plt.plot(kys, np.squeeze(evals)[:,2])
# plt.show()

points = np.pad(np.diff(np.sign(np.diff(np.squeeze(evals)[:,2]))) != 0, 1, constant_values = False)
K_splits = kys[points]
E_splits = np.squeeze(evals)[points, 2]


#plot
plt.plot(kys, np.squeeze(evals)[:,2], label = "D = 1V/nm")

#measure styles
gap_style = {"capsize":3, "elinewidth":1, "markeredgewidth":1, "color": "black"}

#energy gap left
plt.errorbar([K_splits[0],], sum([E_splits[0], E_splits[1]])/2,sum([E_splits[0], -E_splits[1]])/2 , **gap_style)
plt.annotate(str(abs(sum([E_splits[0], -E_splits[1]])))[:6], (K_splits[0],sum([E_splits[0], E_splits[1]])/2))

#energy gap right
plt.errorbar([K_splits[2],], sum([E_splits[2], E_splits[1]])/2,sum([E_splits[2], -E_splits[1]])/2 ,  **gap_style)
plt.annotate(str(abs(sum([E_splits[2], -E_splits[1]])))[:6], (K_splits[2],sum([E_splits[2], E_splits[1]])/2))

#kgap right
plt.errorbar(sum([K_splits[2],K_splits[1]])/2,[0.052,],xerr = sum([K_splits[2], -K_splits[1]])/2 ,  **gap_style)
plt.annotate(str(abs(sum([K_splits[2], -K_splits[1]])))[:6], (sum([K_splits[2],K_splits[1]])/2,0.052))

plt.errorbar(sum([K_splits[0],K_splits[1]])/2,[0.052,],xerr = sum([K_splits[0], -K_splits[1]])/2 ,  **gap_style)
plt.annotate(str(abs(sum([K_splits[0], -K_splits[1]])))[:6], (sum([K_splits[0],K_splits[1]])/2,0.052))
plt.ylim(0.0445,0.0541)
plt.xlim(4.82, 5.07)
plt.xlabel("$k_y$ nm$^{-1}$")
plt.ylabel("Energy $e$V")
plt.savefig("bands.pdf")

# plt.imshow(evals[:,:,2], vmax = 0.1)
# plt.colorbar()
# plt.show()
