import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve 
from functools import reduce
N = 101
A = 0.5/np.pi
ks = np.linspace(-2,2, N)
kxs,kys = np.meshgrid(ks,ks)

e_func = kxs**2 + 0.5*kys**4
mu = 1
vs = plt.contour(ks,ks,e_func, levels=[mu]).collections[0].get_paths()[0].vertices
# plt.show()
# theta = np.linspace(0,2*np.pi, N)
# vs = np.stack((np.cos(theta), np.sin(theta))).T

lens = np.sqrt(np.sum(np.diff(vs, axis=0)**2, axis=1))
# lens = 0.5*(lens + np.roll(lens,1))
mid_points = 0.5*(vs[:-1,:] + vs[1:,:])
mid_points = vs[:-1,:]

def V_kk(k,kk):
    return  (np.tensordot(k,kk, axes =(1,1)))**2

# plt.clf()
# plt.plot(V_kk(mid_points, mid_points)[:,0])
# plt.show()
def rhs(delta):
    return np.sum(V_kk(mid_points, mid_points)*(delta*lens/np.sqrt(np.sum(mid_points**2,axis =1) + delta**2))[np.newaxis,:], axis = 1)


# print(rhs(1*np.ones_like(lens)))
# print(np.sum(lens))
# exit()
# res = fsolve(lambda x: x-rhs(x),10*np.ones_like(lens))
def composite_function(*func):
      
    def compose(f, g):
        return lambda x : f(g(x))
              
    return reduce(compose, func, lambda x : x)
res = composite_function(*(10*(rhs,)))(-0.01*np.ones_like(lens))
plt.clf()
plt.plot(res)
plt.show()
# print(res - rhs(res))
print(np.mean(res), np.std(res))