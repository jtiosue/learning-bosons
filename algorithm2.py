import numpy as np
from algorithm1 import findV
from methods import random_unitary

__all__ = ("findVFock",)


def findVFock(sigma1, sigma2):
    # sigma1, sigma2 come from the state W|f> for some unknown W and f.
    # return V, g such that <f|W^dag V|g> = 1.

    sigma1, sigma2 = sigma1.astype(np.complex128), sigma2.astype(np.complex128)

    n, c = sigma1.shape
    if n != c:
        raise ValueError("sigma is not square or not n^2 by n^2")

    PW = sigma1 - np.eye(n)
    values, U = np.linalg.eigh(PW)
    # if np.any(values.round() < 0):
    #     print(values)
    g = [max(0, int(round(x))) for x in values]  # increasing order of photon number

    Udag = U.conj().T

    sigma2 = np.kron(Udag, Udag) @ sigma2 @ np.kron(U, U)
    V = np.eye(n, dtype=sigma2.dtype)
    index, prev_val, num = 0, g[0], 1
    for f in g[1:]:
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
            Vp = findV(sigma2temp, prev_val)

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
    Vp = findV(sigma2temp, prev_val)

    for i in range(num):
        for j in range(num):
            # V[n - num + i, n - num + j] = Vp[i, j]
            V[index + i, index + j] = Vp[i, j]

    return U @ V, g
