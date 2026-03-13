from .fockgaussian import matelem
from thewalrus.decompositions import blochmessiah
import numpy as np

### TRY NUMBA: https://stackoverflow.com/questions/64609501/python-how-to-make-the-numba-based-for-loop-faster


def sample_heterodyne(S, f, nsamples, initial_anneal=500, Delta=10):
    n = len(f)
    f = list(f)
    # S = S.T
    A, lam, Ap = blochmessiah(S)
    lam = np.diagonal(lam)[:n]
    lam = np.log(lam)
    # lam *=-1
    U = A[:n, :n] + 1j * A[:n, n:]
    Up = Ap[:n, :n] + 1j * Ap[:n, n:]

    # heuristic
    variance = np.linalg.matrix_norm(S) * (1 + 2 * np.mean(f)) / 4

    def probability(alpha):
        # Returns |<alpha| U_S |f>|^2
        return np.abs(matelem(n, [0] * n, f, U, Up, lam, -alpha)) ** 2
        # test the function quickly by using the next line instead of the above line
        # for heterodyne sampling from the vacuum state
        # return np.exp(-np.sum(np.abs(alpha) ** 2)) / (np.pi)**n

    samples = np.zeros((nsamples, n), dtype=np.complex128)
    saved = 0
    alpha = np.random.normal(0, variance, size=n) * np.exp(
        1j * 2 * np.pi * np.random.random(size=n)
    )
    current_pr = probability(alpha)
    accept = 0
    for step in range(initial_anneal + Delta * nsamples):
        newalpha = (
            alpha
            + np.random.normal(0, variance / 2, size=n)
            + 1j * np.random.normal(0, variance / 2, size=n)
        )
        new_pr = probability(newalpha)

        if np.random.random() < new_pr / current_pr:
            alpha = newalpha
            current_pr = new_pr
            accept += 1

        if step >= initial_anneal and not (step - initial_anneal) % Delta:
            samples[saved] = alpha.copy()
            saved += 1

    if nsamples != saved:
        raise ValueError("Something weird happened")
    # print("accept:", accept / (initial_anneal + Delta * nsamples), "%")
    return samples
