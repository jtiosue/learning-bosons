import numpy as np
from .anneal_wrapper import sample_heterodyne
from thewalrus.decompositions import blochmessiah


def heterodyne_samples_from_Gaussian_Fock(f, S, nsamples, initial_anneal=500, Delta=10):
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
    initial_alpha = np.random.normal(0, 1, size=n) + 1.0j * np.random.normal(
        0, 1, size=n
    )

    return sample_heterodyne(
        n, f, U, Up, ls, nsamples, initial_anneal, Delta, initial_alpha
    )
