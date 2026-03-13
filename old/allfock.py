import numpy as np
from thewalrus import perm
from scipy.stats import unitary_group, ortho_group
from math import factorial


def get_linearly_independent_subset(vectors):
    # return q in the qr decomp
    return np.linalg.qr(vectors)[0]


def schmidt_decomp(v, tol=1e-6):
    # with v a n**2 dim vector, return a list of the n-dimensional Schmidt vectors
    # correpsonding to Schmidt coefficients that are greater than the tolererance tol.
    n = int(np.sqrt(len(v)))
    M = v.reshape((n, n))
    U, S, Vh = np.linalg.svd(M)
    return [U[:, i] for i in range(n) if S[i] > tol]


def find_V(sigma2, photon_number):
    # sigma2 comes from the state W|b^n> where b = photon_number
    # return a V such that <b^n|W^dag V|v^n> = 1.
    r, c = sigma2.shape
    n = int(np.sqrt(r))
    if r != c or np.sqrt(r) != n:
        raise ValueError("sigma2 is not square or not n^2 by n^2")

    if not photon_number:
        return np.eye(n, dtype=sigma2.dtype)

    B = np.zeros((n, n, n, n))
    for i in range(n):
        for j in range(n):
            B[i, j, i, j] += (photon_number + 1) ** 2
            B[i, j, j, i] += (photon_number + 1) ** 2

    B = B.reshape((n**2, n**2))

    A = (B - sigma2) / (photon_number * (photon_number + 1))
    # A = (A + A.T) / 2.0

    values, vectors = np.linalg.eigh(A)

    # values is sorted in increasing order

    # this might fail with noise
    # if not np.allclose([1.0] * n, values[n**2 - n :]):
    #     raise ValueError("Not the correct eigenvalues")

    allws = []
    for i in range(n**2 - n, n**2):
        tildew = vectors[:, i]
        allws.extend(schmidt_decomp(tildew))

    allws = np.array(allws).T

    return get_linearly_independent_subset(allws)


def find_V_fock(sigma1, sigma2):
    # sigma1, sigma2 come from the state W|f> for some unknown W and f.
    # return V, g such that <f|W^dag V|g> = 1.

    sigma1, sigma2 = sigma1.astype(np.complex128), sigma2.astype(np.complex128)

    n, c = sigma1.shape
    if n != c:
        raise ValueError("sigma is not square or not n^2 by n^2")

    PW = sigma1 - np.eye(n)
    values, U = np.linalg.eigh(PW)
    fock = [int(round(x)) for x in values]  # increasing order of photon number

    Udag = U.conj().T

    sigma2 = np.kron(Udag, Udag) @ sigma2 @ np.kron(U, U)
    V = np.eye(n, dtype=sigma2.dtype)
    index, prev_val, num = 0, fock[0], 1
    for f in fock[1:]:
        if f == prev_val:
            num += 1
        else:
            sigma2temp = (
                (sigma2.reshape((n, n, n, n)))[
                    index : index + num,
                    index : index + num,
                    index : index + num,
                    index : index + num,
                ]
            ).reshape((num**2, num**2))
            Vp = find_V(sigma2temp, prev_val)

            for i in range(num):
                for j in range(num):
                    # V[n - num + i, n - num + j] = Vp[i, j]
                    V[index + i, index + j] = Vp[i, j]
            index += num
            num = 1
            prev_val = f

    sigma2temp = (
        (sigma2.reshape((n, n, n, n)))[
            index : index + num,
            index : index + num,
            index : index + num,
            index : index + num,
        ]
    ).reshape((num**2, num**2))
    Vp = find_V(sigma2temp, prev_val)

    for i in range(num):
        for j in range(num):
            # V[n - num + i, n - num + j] = Vp[i, j]
            V[index + i, index + j] = Vp[i, j]

    return U @ V, fock


def overlap(U, fock1, fock2):
    # find |<fock1| rho(U) |fock2>|
    # return np.abs(perm(U[:k, :k]))
    if sum(fock1) != sum(fock2):
        return 0.0
    elif len(fock1) == 1:
        return float(fock1[0] == fock2[0])
    elif all(x == 0 for x in fock1) and all(y == 0 for y in fock2):
        return 1.0
    N = sum(fock1)
    rows = []
    for i, f in enumerate(fock1):
        for _ in range(f):
            rows.append(U[i, :])
    rows = np.array(rows)
    M = []
    for i, f in enumerate(fock2):
        for _ in range(f):
            M.append(rows[:, i])
    M = np.array(M).T
    return np.abs(perm(M)) / np.sqrt(
        np.prod([factorial(f) for f in fock1]) * np.prod([factorial(f) for f in fock2])
    )


def create_sigma(W, fock, noise=0):
    # if noise = 0, then we assume that we can measure sigma perfectly
    # if noise > 0, then we add Gaussian noise to our measurements

    r, n = W.shape
    if r != n:
        raise ValueError("W is not square")

    # create sigma
    sigma1 = np.zeros((n, n))
    sigma2 = np.zeros((n, n, n, n))
    for i in range(n):
        sigma1[i, i] = fock[i] + 1

    for i in range(n):
        # sigma2[i, i, i, i] -= (fock[i] + 1) * (fock[i] + 2) - (fock[i] + 1) * (
        #     fock[i] + 1
        # )
        sigma2[i, i, i, i] -= fock[i] * (fock[i] + 1)
        for j in range(n):
            sigma2[i, j, i, j] += (fock[i] + 1) * (fock[j] + 1)
            sigma2[i, j, j, i] += (fock[i] + 1) * (fock[j] + 1)

    sigma2 = sigma2.reshape((n**2, n**2))

    sigma1 = W @ sigma1 @ W.conjugate().T
    WW = np.kron(W, W)
    sigma2 = WW @ sigma2 @ WW.conjugate().T

    # sigma1 = (sigma1 + sigma1.T) / 2.0
    # sigma2 = (sigma2 + sigma2.T) / 2.0

    # sigma1, sigma2 = sigma1.real, sigma2.real

    if noise:
        sigma1 += (2 * np.random.random((n, n)) - 1.0) * noise
        sigma2 += (2 * np.random.random((n**2, n**2)) - 1.0) * noise
        # sigma += np.ones((n**2, n**2)) * noise
        # sigma = (sigma + sigma.conj().T) / 2.0

    return sigma1, sigma2


def check(W, fock, noise=0):
    # if noise = 0, then we assume that we can measure sigma perfectly
    # if noise > 0, then we add noise to our measurements
    sigma1, sigma2 = create_sigma(W, fock, noise)
    V, fock1 = find_V_fock(sigma1, sigma2)

    return overlap(W.conjugate().T @ V, fock, fock1)


def errors(n, fock, noise=0, iters=100, real=False):
    # return the differences between 1 and
    # the magnitude of the overlap that we achieve
    if not real:
        res = [check(unitary_group.rvs(n), fock, noise) for _ in range(iters)]
    else:
        res = [check(ortho_group.rvs(n), fock, noise) for _ in range(iters)]
    return [1 - x for x in res]


def test(n, fock, noise=0, iters=100, real=False):
    # return the maximum difference between 1 and
    # the magnitude of the overlap that we achieve
    return max(errors(n, fock, noise, iters, real))


if __name__ == "__main__":

    import random

    res = []

    for n in range(5, 10):
        print("Starting", n)
        for _ in range(3):
            f = [random.randint(0, 3) for __ in range(n)]
            res.append(test(n, f, real=False))
    print(max(res))
