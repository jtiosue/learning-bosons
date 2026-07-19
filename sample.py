import numpy as np
from methods import fockstate_Lambda


def _sample_heterodyne_fock1(nmodes: int, nsamples: int):
    rng = np.random.default_rng()
    size = nmodes, nsamples
    s = rng.gamma(shape=2.0, scale=1.0, size=size)
    theta = rng.uniform(0, 2 * np.pi, size=size)
    return np.sqrt(s) * np.exp(1.0j * theta)


def sample_heterodyne_passive_fock1(U: np.ndarray, nsamples: int):
    """
    Return heteorodyne samples from a passive linear optical circuit with unitary U,
    starting from a Fock state with 1 photon in each mode.
    """
    n = len(U)
    samples = U @ _sample_heterodyne_fock1(n, nsamples)
    return samples.T


def sample_heterodyne_passive_fock(U: np.ndarray, f: list[int], nsamples: int):
    """
    Return heteorodyne samples from a passive linear optical circuit with unitary U,
    starting from a Fock state with f[i] photons in each mode i.

    This is a straightforward extension of sample_heterodyne_passive_fock1.
    Just need to use different gamma distributions for each mode, and then combine them into a single sample.
    """
    raise NotImplementedError("sample_heterodyne_passive_fock is not yet implemented.")


def sample_heterodyne_Gaussian_fock(S: np.ndarray, f: list[int], nsamples: int):
    """
    Return heteorodyne samples S|f> for an aribtrary Gaussian specified by the symplectic
    matrix S and fock state |f>.

    This function is less straightforward than sample_heterodyne_passive_fock
    """
    raise NotImplementedError("sample_heterodyne_Gaussian_fock is not yet implemented.")
