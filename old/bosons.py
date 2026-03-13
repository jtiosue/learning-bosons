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


def find_V(sigma):
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


def overlap(U):
    # find |<1^n| rho(U) |1^n>|
    return np.abs(perm(U))


def create_sigma(W, noise=0):
    # if noise = 0, then we assume that we can measure sigma perfectly
    # if noise > 0, then we add Gaussian noise to our measurements

    r, n = W.shape
    if r != n:
        raise ValueError("W is not square")

    # create sigma
    sigma = np.zeros((n, n, n, n))
    for i in range(n):
        sigma[i, i, i, i] -= 2
        for j in range(n):
            sigma[i, j, i, j] += 4
            sigma[i, j, j, i] += 4

    sigma = sigma.reshape((n**2, n**2))
    WW = np.kron(W, W)
    sigma = WW @ sigma @ WW.conjugate().T

    if noise:
        sigma += (2 * np.random.random((n**2, n**2)) - 1.0) * noise
        # sigma += np.ones((n**2, n**2)) * noise
        # sigma = (sigma + sigma.conj().T) / 2.0

    return sigma


def check(W, noise=0):
    # if noise = 0, then we assume that we can measure sigma perfectly
    # if noise > 0, then we add noise to our measurements
    sigma = create_sigma(W, noise)
    V = find_V(sigma)

    return overlap(W.conjugate().T @ V)


def errors(n, noise=0, iters=100):
    # return the differences between 1 and
    # the magnitude of the overlap that we achieve
    res = [check(unitary_group.rvs(n), noise) for _ in range(iters)]
    return [1 - x for x in res]


def test(n, noise=0, iters=100):
    # return the maximum difference between 1 and
    # the magnitude of the overlap that we achieve
    return max(errors(n, noise, iters))


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

    c = 1
    alpha = 1
    iters = 1000
    for n in range(5, 11):
        # noise = (1 / np.sqrt(20)) * n ** (-4 - alpha)
        noise = (1 / np.sqrt(20)) * n ** (-4 - alpha - 1 / 2)
        tol = (
            c * n ** (-alpha)
            + 2 * c**2 * n ** (-2 * alpha)
            + 6 * c**3 * n ** (1 - 3 * alpha)
        )  ## require the overlap to be > 1 - tol
        print(
            "Testing with n = %d with %d random unitaries with noise %.1g. desired tolerance %.2g"
            % (n, iters, noise, tol)
        )
        res = test(n, noise=noise, iters=iters)
        print("\tIt worked!" if res < tol else "\tit failed :(")
        print("\tAchieved tolerance of %.2g" % res)
