import numpy as np
from .anneal_wrapper import sample_heterodyne
from thewalrus.decompositions import blochmessiah


def heterodyne_samples_from_Gaussian_Fock(
    f, S, nsamples, stepsize=0.1, initial_anneal=500, Delta=10
):
    """
    return heterodyne samples from the state U_S |f>
    initial_anneal is how long to anneal before beginning to sample
    from the distribution. Delta is the number of steps between grabbing
    samples.
    """
    n = len(f)
    f = list(f)
    A, lam, Ap = blochmessiah(S)
    lam = np.diagonal(lam)[:n]
    ls = np.log(lam)
    # lam *=-1
    U = A[:n, :n] + 1j * A[n:, :n]
    Up = Ap[:n, :n] + 1j * Ap[n:, :n]
    initial_alpha = np.random.normal(0, 2, size=n) + 1.0j * np.random.normal(
        0, 2, size=n
    )
    return sample_heterodyne(
        n, f, U, Up, ls, nsamples, stepsize, initial_anneal, Delta, initial_alpha
    )


def heterodyne_samples_from_vacuum(n, nsamples):
    # p(alpha) = e^{-|alpha|^2}/pi = |<alpha|0>|^2
    real = np.random.multivariate_normal([0] * n, np.diag([1 / 2] * n), nsamples)
    im = np.random.multivariate_normal([0] * n, np.diag([1 / 2] * n), nsamples)
    return real + 1j * im
