from bbg import BLG_H
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvalsh
from functools import reduce
from scipy.optimize import fsolve 

N_k = 401
a0=0.246
Kmax = 0.025/a0*2
kxs = np.linspace(-Kmax,Kmax,N_k) + 2*np.pi/(3*a0)
kys = np.linspace(-Kmax,Kmax,N_k) + 2*np.pi/(3*np.sqrt(3)*a0)

H = BLG_H([kxs, kys], D=1.0)
evals = eigvalsh(H)

ws = np.linspace(0.045, 0.055, 15)
ws = list(ws) + [1,]
plt.contourf(kxs,kys, evals[:,:,2], levels = ws, vmax = 0.055, vmin = 0.045, cmap = "hot")
plt.colorbar()
plt.gca().set_aspect(1)
plt.xlabel("$k_x$ nm$^{-1}$")
plt.ylabel("$k_y$ nm$^{-1}$")


#now generate the contour for integrating
vs = plt.contour(kxs,kys,evals[:,:,2], levels=[0.05], linewidths=0.0).collections[0].get_paths()[0].vertices
lens = np.sqrt(np.sum(np.diff(vs, axis=0)**2, axis=1))
mid_points = 0.5*(vs[:-1,:] + vs[1:,:])
mid_points = vs[:-1,:]
plt.scatter(mid_points[::5,0], mid_points[::5,1],marker = "x", s=1)
plt.savefig("sampled_for_integral.pdf")
plt.clf()


#define the V_kk'
def V_kk(k,kk):
    return  (np.tensordot(k,kk, axes =(1,1)))**2

#define the integral
def rhs(delta):
    return np.sum(V_kk(mid_points, mid_points)*(delta*lens/np.sqrt(np.sum(mid_points**2,axis =1) + delta**2))[np.newaxis,:], axis = 1)


def composite_function(*func):    
    def compose(f, g):
        return lambda x : f(g(x))  
    return reduce(compose, func, lambda x : x)

from mpl_toolkits.mplot3d import Axes3D


res = 7500*np.ones_like(lens)

steps = []
for iter in range(5):
    
    steps.append(res)
    res = rhs(res)


from scipy.interpolate import interp1d as li

ts = np.linspace(0,1, num = len(steps))
samples = list(map(lambda x: li(ts, x), [[x[i] for x in steps] for i in range(res.size)]))
t_fine =  np.linspace(0,0.3, num = 20)

fig = plt.figure()
plt.clf()
plt.ion()
for i,t in enumerate(t_fine):
    z = [float(f(t)) for f in samples]
    ax = fig.add_subplot(111, projection="3d")
    ax.set_zlim(7000,8000)
    ax.set_xlabel("$k_x$ nm$^{-1}$")
    ax.set_ylabel("$k_y$ nm$^{-1}$")
    ax.set_zlabel("$\Delta_k$")
    ax.set_zticklabels([])
    ax.plot(mid_points[:,0], mid_points[:,1], z)
    fig.savefig(f"animation/frame_{i}.png")
    fig.clf()



    
res = composite_function(*(10*(rhs,)))(-0.01*np.ones_like(lens))
plt.clf()
plt.plot(res)
plt.show()
print(np.mean(res - rhs(res)))
print(np.mean(res), np.std(res))