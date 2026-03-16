import numpy as np
from thewalrus.random import random_symplectic
from algorithm1 import findV
from algorithm2 import findVFock
from algorithm3 import findQ
from methods import (
    sigma_from_Lambda,
    random_unitary,
    fockstate_sigma,
    fockstate_Lambda,
    estimate_Lambda_from_samples,
)
from heterodyne import (
    passive_overlap,
    Gaussian_overlap,
    heterodyne_samples_from_Gaussian_Fock,
)

ATOL = 0.2
ASSERT_TESTS = False


def testing_wrapper(fun):
    def newfun(*args, **kwargs):
        print("Testing", fun.__name__)
        print()
        fun(*args, **kwargs)
        print(fun.__name__, "passed")
        print()

    return newfun


@testing_wrapper
def test_constfock_passive(n, b, nsamples):
    f = (b,) * n
    W = random_unitary(n)
    W = np.array(
        [
            [
                (-0.5786025203672942 - 0.7833624406506234j),
                (0.12927082250493085 - 0.0260176124213277j),
                (0.1701903348130576 - 0.07217893204754894j),
            ],
            [
                (0.16942677999633324 + 0.12209876854821916j),
                (0.30918077381166875 - 0.538191312918597j),
                (0.6945928659232777 - 0.2977995422289031j),
            ],
            [
                (0.003950360466120066 - 0.08906885872056142j),
                (0.17501919196911328 - 0.7528198858554714j),
                (-0.46576267496969775 + 0.42160024379322836j),
            ],
        ]
    )
    WW = np.kron(W, W)
    S = np.block([[W.real, W.imag], [-W.imag, W.real]])
    SS = np.kron(S, S)

    samples = heterodyne_samples_from_Gaussian_Fock(f, S, nsamples)

    Lambda1, Lambda2 = estimate_Lambda_from_samples(samples)
    sigma1, sigma2 = sigma_from_Lambda(Lambda1, Lambda2)

    # with enough samples, these should all pass. But it might require a lot of samples
    if ASSERT_TESTS:
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

    V = findV(sigma2, b)
    Q = np.block([[V.real, V.imag], [-V.imag, V.real]])
    print(Gaussian_overlap(f, np.linalg.inv(S) @ Q, f))
    print(passive_overlap(f, W.conj().T @ V, f))


@testing_wrapper
def test_passive(f, nsamples):
    n = len(f)
    W = random_unitary(n)
    WW = np.kron(W, W)
    S = np.block([[W.real, W.imag], [-W.imag, W.real]])
    SS = np.kron(S, S)

    samples = heterodyne_samples_from_Gaussian_Fock(f, S, nsamples)

    Lambda1, Lambda2 = estimate_Lambda_from_samples(samples)
    sigma1, sigma2 = sigma_from_Lambda(Lambda1, Lambda2)

    # with enough samples, these should all pass. But it might require a lot of samples
    if ASSERT_TESTS:
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

    Q = np.block([[V.real, V.imag], [-V.imag, V.real]])
    print(Gaussian_overlap(f, np.linalg.inv(S) @ Q, g))


@testing_wrapper
def test_Gaussian(f, nsamples):
    n = len(f)
    S = random_symplectic(n, scale=0.5)
    SS = np.kron(S, S)

    samples = heterodyne_samples_from_Gaussian_Fock(f, S, nsamples)

    Lambda1, Lambda2 = estimate_Lambda_from_samples(samples)

    # with enough samples, these should all pass. But it might require a lot of samples
    if ASSERT_TESTS:
        Lambda1_init, Lambda2_init = fockstate_Lambda(f)
        np.testing.assert_allclose(
            Lambda1, S @ Lambda1_init @ S.T, atol=ATOL, verbose=False
        )
        np.testing.assert_allclose(
            Lambda2, SS @ Lambda2_init @ SS.T, atol=ATOL, verbose=False
        )

    Q, g = findQ(Lambda1, Lambda2)
    print(Gaussian_overlap(f, np.linalg.inv(S) @ Q, g))


@testing_wrapper
def test_Gaussian_vacuum(n, nsamples):
    S = random_symplectic(n, scale=0.5)
    SS = np.kron(S, S)

    samples = heterodyne_samples_from_Gaussian_Fock([0] * n, S, nsamples)

    Lambda1, Lambda2 = estimate_Lambda_from_samples(samples)
    if ASSERT_TESTS:
        Lambda1_init, Lambda2_init = fockstate_Lambda((0,) * n)
        np.testing.assert_allclose(
            Lambda1, S @ Lambda1_init @ S.T, atol=ATOL, verbose=False
        )
        np.testing.assert_allclose(
            Lambda2, SS @ Lambda2_init @ SS.T, atol=ATOL, verbose=False
        )

    Q, g = findQ(Lambda1, Lambda2)
    print(Gaussian_overlap([0] * n, np.linalg.inv(S) @ Q, g))


if __name__ == "__main__":
    n = 3
    nsamples = 500000
    # f = np.random.randint(0, 3, n)
    # test_vacuum(n, nsamples)
    test_constfock_passive(n, 1, nsamples)
    # test_passive((1, 1), nsamples)
    # test_Gaussian(f, nsamples)
    # test_Gaussian_vacuum(n, nsamples)
