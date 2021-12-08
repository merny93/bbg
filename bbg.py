import numpy as np
from matplotlib.pyplot import contour
from functools import reduce
import matplotlib.pyplot as plt

from time import perf_counter as tt


Params = dict({'g0':-2.61, 'g1':0.361 , 'g3':0.283, 'g4':0.138, 'Dp':0.015})

def DOSk(w, Es, eta):
    """ Computes spectral functino A(k;w) = -ImTrG(k;w) / pi
    
        Es: (N_band, N_kx, N_ky)
        eta: Lorentzian broadening parameter (energy resolution, should be chosen wisely depending on the k-grid resolution)
    
    """
    DOSk = -np.imag( np.sum( 1/(w + 1j*eta -Es), axis=0 ) )/np.pi
    return DOSk

def DOS(ws, Es, eta):
    """ Computes density of states rho(w) = -ImTr\sum_{k}G(k;w)/pi
    
        Es: (N_band, N_kx, N_ky)
        eta: Lorentzian broadening parameter (energy resolution, should be chosen wisely depending on the k-grid resolution)
    
    """
    res = np.zeros(ws.size)
    for i, w in enumerate(ws):
        res[i] = -np.imag( np.sum( 1/(w + 1j*eta -Es) ) )/np.pi
    return res

def DOS_fast(ws, Es):
    """compute density of state by integrating the fermi surfaces to zeroth order
    Input:
        ws (1d array): energies to evaluate DOS at
        Es (2d array): bands to compute DOS from
    Returns: 
        DOS (1d array): DOS computed at ws
    """
    def arc_length(path):
        def length_worker(pp):
            dxdy = np.diff(pp, axis=0)
            dx2dy2 = dxdy**2
            arcs2 = np.sum(dx2dy2, axis = 1)
            arcs = np.sqrt(arcs2)
            arc = np.sum(arcs)
            return arc
        return reduce(lambda x,y: x+length_worker(y.vertices), path,0.0)
    contour_plot = contour(Es, levels = ws)
    cs = contour_plot.collections
    res = np.zeros(ws.size)
    for i, c in enumerate(cs):
        paths = c.get_paths()
        l = arc_length(paths)
        res[i] = l
    return res

def DOSk_fast(ws, Es):
    """Compute FS at set of energies
    Input:
        ws (1d iterable): energies to evaluate at
        Es (2d array): bands to compute FS from
    Returns:
        FS ()
    """       

def BLG_H(q, Params = Params, D = 0, B=0):
    #constants
    a0 = 0.246 #[nm]
    d = 0.34 #[nm]
    epsilon = 3.4 #may vary between 3-8 for BLG
    U = D*d/epsilon
    #hbar = 1

    delta = a0*np.array([[1/2, np.sqrt(3)/2],[1/2, -np.sqrt(3)/2], [-1, 0]])
    
    kx,ky = np.meshgrid(*q)
    k = np.stack((kx,ky))
    f_k = np.sum(np.exp(-1j*np.tensordot(k, delta, axes=(0,1))), axis = -1 )
    F_k = np.abs(f_k)**2 - 3
    
    p = Params
    (t, t0, t3, t4, Dp) = (p['g0'], p['g1'], p['g3'], p['g4'], p['Dp']) #[eV]
    tp = 0
    #g0: in-plane NN A-B / A'-B'
    #g1: interlayer NN B-A'
    #g3: interlayer triagonal warping A-B'
    #g4: interlayer P-H breaking A-A'/B-B'
    #Dp: sublattice-polarizing mass term
    #included g4,Dp for completeness (as done in Sankar's paper Eq.S1)
    h_k_t = np.zeros((4,4))
    h_k_tp = np.zeros((4,4))
    h_k_t0 = np.zeros((4,4))
    h_k_t3 = np.zeros((4,4))
    h_k_t4 = np.zeros((4,4))
    h_v = np.zeros((4,4))

    h_k_t[0,2] = -t
    h_k_t[1,3] = -t

    h_k_tp = -tp * np.eye(4) *0.5 #so we can add in the conjugate

    h_k_t0[1,2] = -t0

    h_k_t3[3,0] = -t3

    h_k_t4[3,2] = -t4

    h_v = U/2*np.diag([1,-1,1,-1]) *0.5 # for hermition

    H_mats = [h_k_t,h_k_tp, h_k_t0, h_k_t3, h_k_t4, h_v]
    mults = [f_k, F_k, np.ones_like(f_k), f_k, f_k, np.ones_like(f_k)]

    h_mults = sum(map(lambda x: np.multiply.outer(*x), zip(mults, H_mats)))
    H = h_mults + h_mults.conjugate().transpose((0,1,3,2))
    return H
    
"""
     Linearized BLG Hamiltonian at valley K in the basis (phi_A, phi_B', phi_B, phi_A'), under out-of-plane electric field (U) and in-plane magnetic field (B)  
"""
def BLG_lin_K(q, Params, D, B=0, theta=0, zeeman=None):
    """
        q: momentum [1/a0], tuple of np.array
        Params: dictionary of tight-binding parameters [eV]
        D: displacement field [V/nm]
        B: magnetic field [T]
        theta: 0 <= theta <= 2*pi
        zeeman: None / 'up' / 'down'
    """
    #constants
    a0 = 0.246 #[nm]
    d = 0.34 #[nm]
    epsilon = 3.4 #may vary between 3-8 for BLG
    #hbar = 1
    
    #momentum
    (kx, ky) = q #[1/a0]
    qp = np.add.outer(kx,1j*ky) #linearization
    qm = np.add.outer(kx,-1j*ky)
    
    ###estimation of electric potential
    #U = D*d/\epsilon; D=1V/nm, d=0.34nm, \epsilon=3.4 -> U(D=1V/nm) ~ 100 meV
    U = D*d/epsilon
    
    ###estimation of orbital-magnetic effect (momentum shift due to Pierel substitution)
    #Bvec = B (cos(theta), sin(theta), 0) -> Avec = zvec cross Bvec
    #c0 = e*Tesla/(2*hbar) = 7.596*10^-4 nm^-2
    #kBshift(in unit of [1/a0]) = \pm c0 d/2 B(in unit of [T])
    #= (-1)^(layer) d/2 * c0 * B * 0.246 * (sin(theta), -cos(theta), 0)
    pfac = 0.34/2*7.596*10**(-4)*B*0.246 # pfac~5*10^-6/a0 with B=0.165T, very small
    qpt = np.add.outer((kx + pfac*np.sin(theta)), 1j * (ky - pfac*np.cos(theta)))
    qpb = np.add.outer((kx + pfac*np.sin(theta)), 1j * (ky - pfac*np.cos(theta)))
    qmt = np.add.outer((kx - pfac*np.sin(theta)), -1j * (ky + pfac*np.cos(theta)))
    qmb = np.add.outer((kx - pfac*np.sin(theta)), -1j * (ky + pfac*np.cos(theta)))
    
    #tight-binding parameters
    p = Params
    (g0, g1, g3, g4, Dp) = (p['g0'], p['g1'], p['g3'], p['g4'], p['Dp']) #[eV]
    #g0: in-plane NN A-B / A'-B'
    #g1: interlayer NN B-A'
    #g3: interlayer triagonal warping A-B'
    #g4: interlayer P-H breaking A-A'/B-B'
    #Dp: sublattice-polarizing mass term
    #included g4,Dp for completeness (as done in Sankar's paper Eq.S1)

    v0 = np.sqrt(3)/2*a0*g0 #per hbar; a0 is needed to cancel 1/a0 factor implicit from momentum q
    v4 = np.sqrt(3)/2*a0*g4
    v3 = np.sqrt(3)/2*a0*g3

    #(A, B', B, A'), (top, bottom, top, bottom)
    H = np.zeros((len(kx),len(ky),4,4), dtype=complex)
    
    H[...,0,0] = U/2
    H[...,0,1] = v3*qm
    H[...,0,2] = v0*qpt
    H[...,0,3] = v4*qp
    
    H[...,1,0] = v3*qp
    H[...,1,1] = -U/2
    H[...,1,2] = v4*qm
    H[...,1,3] = v0*qmb
    
    H[...,2,0] = v0*qmt
    H[...,2,1] = v4*qp
    H[...,2,2] = U/2 + Dp
    H[...,2,3] = g1
    
    H[...,3,0] = v4*qm
    H[...,3,1] = v0*qpb
    H[...,3,2] = g1
    H[...,3,3] = -U/2 + Dp
    
    return H


def get_zeeman_E(B, spin=None):
    muB = 5.7883818012*10**(-5) #[eV/T]
    gs = 2.1 #roughly near 2.0 
    ###estimation of spin-Zeeman
    #Ez = \pm gs muB B/2; gs~2.1, muB~5.7883818012(26)×10−5 eV/T, B=0.165T -> ~\pm 0.01 meV
    if spin is not None:
        Ez = muB*B*gs/2 if spin.lower()=='up' else -muB*B*gs/2
    else:
        Ez = 0.
    return Ez