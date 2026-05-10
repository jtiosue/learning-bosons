import numpy as np
from methods import fockstate_Lambda


def _sample_heterodyne_fock1(nmodes: int, nsamples: int):
    rng = np.random.default_rng()
    size = nmodes, nsamples
    s = rng.gamma(shape=2.0, scale=1.0, size=size)
    theta = rng.uniform(0, 2 * np.pi, size=size)
    return np.sqrt(s) * np.exp(1.0j * theta)


def sample_heterodyne_passive_fock1(U: np.ndarray, nsamples: int):
    n = len(U)
    samples = U @ _sample_heterodyne_fock1(n, nsamples)
    return samples.T


def shadow_sample(f, S, nsamples=1000):
    """
    Sample from a Gaussian distribution around the true matrix entries
    of the Lambdas.

    TODO: make this more correct!
    """
    SS = np.kron(S, S)
    Lambda1_0, Lambda2_0 = fockstate_Lambda(f)
    true_L1 = S @ Lambda1_0 @ S.T
    true_L2 = SS @ Lambda2_0 @ SS.T
    # TODO: make this actually the true_L4
    true_L4 = max(f) * true_L2**2

    l = len(S)
    L1 = np.random.multivariate_normal(
        true_L1.real.reshape((l**2,)), np.eye(l**2) * np.abs(true_L2).max(), nsamples
    ).reshape((nsamples, l, l)) + np.random.multivariate_normal(
        true_L1.imag.reshape((l**2,)), np.eye(l**2) * np.abs(true_L2).max(), nsamples
    ).reshape(
        (nsamples, l, l)
    )
    L2 = np.random.multivariate_normal(
        true_L2.real.reshape((l**4,)), np.eye(l**4) * np.abs(true_L4).max(), nsamples
    ).reshape((nsamples, l**2, l**2)) + np.random.multivariate_normal(
        true_L2.imag.reshape((l**4,)), np.eye(l**4) * np.abs(true_L4).max(), nsamples
    ).reshape(
        (nsamples, l**2, l**2)
    )
    return L1, L2
