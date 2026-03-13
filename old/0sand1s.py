import numpy as np
from thewalrus import perm
from scipy.stats import unitary_group


def get_linearly_independent_subset(vectors):
    Q, R = np.linalg.qr(vectors)
    return Q


def schmidt_decomp(v):
    n = int(np.sqrt(len(v)))
    # Step 1: Reshape the vector into an n x n matrix
    M = v.reshape((n, n))

    # Step 2: Perform singular value decomposition
    U, S, Vh = np.linalg.svd(M)

    # Step 3: Schmidt coefficients are the singular values
    schmidt_coeffs = S

    # Step 4: Corresponding orthonormal bases (columns of U and rows of Vh)
    A_basis = U  # in subsystem A
    # return [U[:, i] for i in range(n)]
    return [U[:, i] for i in range(n) if schmidt_coeffs[i]]

    # B_basis = Vh.conj().T  # in subsystem B

    # # Optional: verify reconstruction
    # reconstructed = sum(s * np.kron(A_basis[:, i], B_basis[:, i]) for i, s in enumerate(S))
    # assert np.allclose(reconstructed, v)

    # return []


def find_smallV(sigma):
    r, c = sigma.shape
    n = int(np.sqrt(r))
    if r != c or np.sqrt(r) != n:
        raise ValueError("sigma is not square or not n^2 by n^2")

    B = np.zeros((n, n, n, n))
    for i in range(n):
        for j in range(n):
            B[i, j, i, j] += 4
            B[i, j, j, i] += 4

    B = B.reshape((n**2, n**2))

    A = (B - sigma) / 2.0
    # A = (A + A.conj().T) / 2.0

    values, vectors = np.linalg.eigh(A)

    # this might fail with noise
    # if not np.allclose([1.0] * n, values[n**2 - n :]):
    #     raise ValueError("Not the correct eigenvalues")

    allws = []
    for i in range(n**2 - n, n**2):
        tildew = vectors[:, i]
        allws.extend(schmidt_decomp(tildew))

    allws = np.array(allws).T

    return get_linearly_independent_subset(allws)


def find_V(sigma1, sigma2, k):
    n, c = sigma1.shape
    if n != c:
        raise ValueError("sigma is not square or not n^2 by n^2")

    PW = sigma1 - np.eye(n)
    values, U = np.linalg.eigh(PW)
    if not np.allclose(values, [0] * (n - k) + [1] * k):
        raise ValueError("Problem")
    U = U[:, ::-1]
    Udag = U.conj().T

    sigma2 = np.kron(Udag, Udag) @ sigma2 @ np.kron(U, U)
    # sigma2 = np.kron(U, U) @ sigma2 @ np.kron(Udag, Udag)
    # sigma2 = (
    #     (sigma2.reshape((n, n, n, n)))[n - k :, n - k :, n - k :, n - k :]
    # ).reshape((k**2, k**2))
    sigma2 = ((sigma2.reshape((n, n, n, n)))[:k, :k, :k, :k]).reshape((k**2, k**2))
    Vp = find_smallV(sigma2)
    V = np.eye(n, dtype=Vp.dtype)
    for i in range(k):
        for j in range(k):
            # V[n - k + i, n - k + j] = Vp[i, j]
            V[i, j] = Vp[i, j]

    return U @ V


def overlap(U, k):
    # find |<1^k 0^{n-k}| rho(U) |1^k 0^{n-k}>|
    return np.abs(perm(U[:k, :k]))


def create_sigma(W, k, noise=0):
    # if noise = 0, then we assume that we can measure sigma perfectly
    # if noise > 0, then we add Gaussian noise to our measurements

    r, n = W.shape
    if r != n:
        raise ValueError("W is not square")

    # create sigma
    sigma1 = np.zeros((n, n))
    sigma2 = np.zeros((n, n, n, n))
    for i in range(k):
        sigma1[i, i] = 2
    for i in range(k, n):
        sigma1[i, i] = 1

    for i in range(n):
        for j in range(n):
            for l in (i, j):
                for m in (i, j):
                    if all((i < k, j < k, l < k, m < k)):
                        if i == j == l == m:
                            sigma2[i, j, l, m] = 6
                        elif {i, j} == {l, m}:
                            sigma2[i, j, l, m] = 4
                    elif all((i >= k, j >= k, l >= k, m >= k)):
                        if i == j == l == m:
                            sigma2[i, j, l, m] = 2
                        elif {i, j} == {l, m}:
                            sigma2[i, j, l, m] = 1
                    elif {i, j} == {l, m}:
                        sigma2[i, j, l, m] = 2

    sigma1 = W @ sigma1 @ W.conjugate().T
    sigma2 = sigma2.reshape((n**2, n**2))
    WW = np.kron(W, W)
    sigma2 = WW @ sigma2 @ WW.conjugate().T

    if noise:
        sigma1 += (2 * np.random.random((n, n)) - 1.0) * noise
        sigma2 += (2 * np.random.random((n**2, n**2)) - 1.0) * noise
        # sigma += np.ones((n**2, n**2)) * noise
        # sigma = (sigma + sigma.conj().T) / 2.0

    return sigma1, sigma2


def check(W, k, noise=0):
    # if noise = 0, then we assume that we can measure sigma perfectly
    # if noise > 0, then we add noise to our measurements
    sigma1, sigma2 = create_sigma(W, k, noise)
    V = find_V(sigma1, sigma2, k)

    return overlap(W.conjugate().T @ V, k)


def errors(n, k, noise=0, iters=100):
    # return the differences between 1 and
    # the magnitude of the overlap that we achieve
    res = [check(unitary_group.rvs(n), k, noise) for _ in range(iters)]
    return [1 - x for x in res]


def test(n, k, noise=0, iters=100):
    # return the maximum difference between 1 and
    # the magnitude of the overlap that we achieve
    return max(errors(n, k, noise, iters))


if __name__ == "__main__":

    # example of W^\dag V. Because V should equal W Phi P,
    # we should see a permutation matrix with phases
    """
    n = 4
    W = unitary_group.rvs(n)
    sigma = create_sigma(W, noise=0)
    V = find_V(sigma)
    Phi_P = W.conj().T @ V
    P = (np.abs(Phi_P) ** 2).round(5)
    Phi = Phi_P @ P.T
    # print(Phi_P.round(3))
    print("P =\n", P, sep="")
    print("Phi =\n", Phi.round(3), sep="")
    print("<1^n| rho(W^\dag V) |1^n> = ", end="")
    print(np.round(perm(Phi_P), 3))
    print("det(Phi)                  = ", end="")
    print(np.round(np.linalg.det(Phi), 3))
    print()

    assert 0
    """

    print(test(15, 12))
