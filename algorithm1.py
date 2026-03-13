import numpy as np

__all__ = ("findV",)


def get_linearly_independent_subset(vectors):
    # return q in the qr decomp
    return np.linalg.qr(vectors)[0]


def schmidt_decomp(v, tol=1e-6):
    # with v a n**2 dim vector, return a list of the n-dimensional Schmidt vectors
    # correpsonding to the nonegenerate Schmidt coefficients that are greater than the tolererance tol.
    n = int(np.sqrt(len(v)))
    M = v.reshape((n, n))
    # S is in descending order
    U, S, Vh = np.linalg.svd(M)

    vecs = []
    for i in range(n):
        if (s := S[i]) < tol:
            continue
        elif i and abs(S[i - 1] - s) < tol:
            continue
        elif i < n - 1 and abs(S[i + 1] - s) < tol:
            continue
        vecs.append(U[:, i])
    return vecs


def findV(sigma2, b):
    # sigma2 comes from the state W|b^n>
    # return a V such that <b^n|W^dag V|v^n> = 1.
    r, c = sigma2.shape
    n = int(np.sqrt(r))
    if r != c or np.sqrt(r) != n:
        raise ValueError("sigma2 is not square or not n^2 by n^2")

    if not b:
        return np.eye(n, dtype=sigma2.dtype)

    B = np.zeros((n, n, n, n))
    for i in range(n):
        for j in range(n):
            B[i, j, i, j] += (b + 1) ** 2
            B[i, j, j, i] += (b + 1) ** 2

    B = B.reshape((n**2, n**2))

    A = (B - sigma2) / (b * (b + 1))
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
