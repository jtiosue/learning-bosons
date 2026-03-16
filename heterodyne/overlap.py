import numpy as np
from .anneal_wrapper import overlap
from thewalrus.decompositions import blochmessiah


def Gaussian_overlap(f, S, g):
    # computes |<f| U_S |g>|
    n = len(f)
    f, g = list(f), list(g)
    A, lam, Ap = blochmessiah(S)
    lam = np.diagonal(lam)[:n]
    ls = np.log(lam)
    # lam *=-1
    U = A[:n, :n] + 1j * A[n:, :n]
    Up = Ap[:n, :n] + 1j * Ap[n:, :n]

    return overlap(n, f, g, U, Up, ls, [0] * n)


def passive_overlap(f, W, g):
    # computes |<f| U_S |g>|
    S = np.block([[W.real, -W.imag], [W.imag, W.real]])
    return Gaussian_overlap(f, S, g)
