import numpy as np
from strawberryfields.utils import random_symplectic
import matplotlib.pyplot as plt
from algorithm1 import findV
from algorithm2 import findVFock
from algorithm3 import findQ
from methods import (
    passive_overlap,
    Gaussian_overlap,
    sigma_from_Lambda,
    random_unitary,
    fockstate_sigma,
    fockstate_Lambda,
)
from heterodyne import (
    estimate_Lambda_from_samples,
    heterodyne_samples_from_passive_Fock,
    heterodyne_samples_from_Gaussian_Fock,
    heterodyne_samples_from_vacuum,
    heterodyne_samples_from_Gaussian,
)
import gpt.heterodyne
import mysim.heterodyne

ATOL = 0.2


def testing_wrapper(fun):
    def newfun(*args, **kwargs):
        print("Testing", fun.__name__)
        print()
        fun(*args, **kwargs)
        print(fun.__name__, "passed")
        print()

    return newfun


@testing_wrapper
def test_vacuum(n, nsamples):
    b = 0
    f = (b,) * n
    samples = heterodyne_samples_from_vacuum(n, nsamples)
    Lambda1, Lambda2 = estimate_Lambda_from_samples(samples)
    sigma1, sigma2 = sigma_from_Lambda(Lambda1, Lambda2)

    Lambda1_init, Lambda2_init = fockstate_Lambda(f)
    sigma1_init, sigma2_init = fockstate_sigma(f)
    np.testing.assert_allclose(sigma1, sigma1_init, atol=ATOL, verbose=False)
    np.testing.assert_allclose(sigma2, sigma2_init, atol=ATOL, verbose=False)
    np.testing.assert_allclose(Lambda1, Lambda1_init, atol=ATOL, verbose=False)
    np.testing.assert_allclose(Lambda2, Lambda2_init, atol=ATOL, verbose=False)


@testing_wrapper
def test_constfock_passive(n, b, nsamples):
    f = (b,) * n
    W = random_unitary(n)
    WW = np.kron(W, W)
    S = np.zeros((2 * n, 2 * n))
    S[:n, :n] = W.real
    S[n:, n:] = W.real
    S[:n, n:] = W.imag
    S[n:, :n] = -W.imag
    SS = np.kron(S, S)

    # samples = heterodyne_samples_from_passive_Fock(f, W, nsamples)
    # samples = gpt.heterodyne.sample_heterodyne(
    #     S,
    #     f,
    #     nsamples,
    #     cutoff=sum(f) + 1,
    #     max_tensor_entries=6_000_000,
    #     # burn_in=1000,
    #     # thinning=200,
    # )
    samples = mysim.heterodyne.sample_heterodyne(
        S,
        f,
        nsamples,
    )

    Lambda1, Lambda2 = estimate_Lambda_from_samples(samples)
    sigma1, sigma2 = sigma_from_Lambda(Lambda1, Lambda2)
    Lambda1_init, Lambda2_init = fockstate_Lambda(f)
    sigma1_init, sigma2_init = fockstate_sigma(f)
    np.testing.assert_allclose(
        sigma1, W @ sigma1_init @ W.conj().T, atol=ATOL, verbose=False
    )
    np.testing.assert_allclose(
        sigma2, WW @ sigma2_init @ WW.conj().T, atol=ATOL, verbose=False
    )
    np.testing.assert_allclose(
        Lambda1, S @ Lambda1_init @ S.T, atol=ATOL, verbose=False
    )
    np.testing.assert_allclose(
        Lambda2, SS @ Lambda2_init @ SS.T, atol=ATOL, verbose=False
    )


@testing_wrapper
def test_passive(f, nsamples):
    n = len(f)
    W = random_unitary(n)
    WW = np.kron(W, W)
    S = np.zeros((2 * n, 2 * n))
    S[:n, :n] = W.real
    S[n:, n:] = W.real
    S[:n, n:] = W.imag
    S[n:, :n] = -W.imag
    SS = np.kron(S, S)

    # samples = heterodyne_samples_from_passive_Fock(f, W, nsamples)
    # samples = gpt.heterodyne.sample_heterodyne(
    #     S, f, nsamples, cutoff=sum(f) + 1, max_tensor_entries=6_000_000
    # )
    samples = mysim.heterodyne.sample_heterodyne(
        S,
        f,
        nsamples,
    )
    # samples = heterodyne_samples_from_passive_Fock(f, W, nsamples)

    Lambda1, Lambda2 = estimate_Lambda_from_samples(samples)
    sigma1, sigma2 = sigma_from_Lambda(Lambda1, Lambda2)
    Lambda1_init, Lambda2_init = fockstate_Lambda(f)
    sigma1_init, sigma2_init = fockstate_sigma(f)
    np.testing.assert_allclose(
        sigma1, W @ sigma1_init @ W.conj().T, atol=ATOL, verbose=False
    )
    np.testing.assert_allclose(
        sigma2, WW @ sigma2_init @ WW.conj().T, atol=ATOL, verbose=False
    )
    np.testing.assert_allclose(
        Lambda1, S @ Lambda1_init @ S.T, atol=ATOL, verbose=False
    )
    np.testing.assert_allclose(
        Lambda2, SS @ Lambda2_init @ SS.T, atol=ATOL, verbose=False
    )
    V, g = findVFock(sigma1, sigma2)
    print(passive_overlap(g, V.conj().T @ W, f))


@testing_wrapper
def test_Gaussian(f, nsamples):
    n = len(f)
    S = random_symplectic(n, scale=0.1)
    SS = np.kron(S, S)

    # samples = heterodyne_samples_from_Gaussian_Fock(f, S, nsamples)
    # samples = gpt.heterodyne.sample_heterodyne(
    #     S, f, nsamples, max_tensor_entries=6_000_000
    # )
    samples = mysim.heterodyne.sample_heterodyne(
        S,
        f,
        nsamples,
    )

    Lambda1, Lambda2 = estimate_Lambda_from_samples(samples)
    Lambda1_init, Lambda2_init = fockstate_Lambda(f)
    np.testing.assert_allclose(
        Lambda1, S @ Lambda1_init @ S.T, atol=ATOL, verbose=False
    )
    np.testing.assert_allclose(
        Lambda2, SS @ Lambda2_init @ SS.T, atol=ATOL, verbose=False
    )


@testing_wrapper
def test_Gaussian_vacuum(n, nsamples):
    S = random_symplectic(n, scale=0.1)
    SS = np.kron(S, S)

    samples = heterodyne_samples_from_Gaussian(S, nsamples)

    Lambda1, Lambda2 = estimate_Lambda_from_samples(samples)
    Lambda1_init, Lambda2_init = fockstate_Lambda((0,) * n)
    np.testing.assert_allclose(
        Lambda1, S @ Lambda1_init @ S.T, atol=ATOL, verbose=False
    )
    np.testing.assert_allclose(
        Lambda2, SS @ Lambda2_init @ SS.T, atol=ATOL, verbose=False
    )


if __name__ == "__main__":
    n = 2
    nsamples = 100000
    # f = np.random.randint(0, 3, n)
    # test_vacuum(n, nsamples)
    # test_constfock_passive(n, 1, nsamples)
    test_passive((1, 2, 1), nsamples)
    # test_Gaussian(f, nsamples)
    # test_Gaussian_vacuum(n, nsamples)
