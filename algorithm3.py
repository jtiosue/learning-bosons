import numpy as np
from thewalrus.decompositions import williamson
from methods import sigma_from_Lambda, create_symplectic_from_unitary
from algorithm2 import findVFock


def findQ(Lambda1, Lambda2):
    # Lambda1, Lambda2 come from the state \mathcal{U}_S|f> for some unknown S and f.
    # return Q, g such that <f|\mathcalU_S^\dag \mathcalU_Q|g> = 1.
    n = len(Lambda1) // 2

    P, R = williamson((Lambda1.real + Lambda1.real.T) / 2.0)
    f = P.diagonal() - 1 / 2
    f = f[: len(f) // 2]
    f.sort()

    Rinv = np.linalg.inv(R)

    Lambda1 = Rinv @ Lambda1 @ Rinv.T
    Lambda2 = np.kron(Rinv, Rinv) @ Lambda2 @ np.kron(Rinv.T, Rinv.T)

    sigma1, sigma2 = sigma_from_Lambda(Lambda1, Lambda2)

    V, g = findVFock(sigma1, sigma2)

    # this may fail when there is enough noise
    # if not np.allclose(f, g):
    #     raise ValueError(f"g != f; g = {g}, f = {f}")

    # if not np.allclose(f.round(), g):
    # raise ValueError(f"g != f; g = {g}, f = {f}")

    Op = create_symplectic_from_unitary(V)

    return R @ Op, g
