import numpy as np
from scipy.stats import unitary_group, ortho_group
import scipy, random
from thewalrus.decompositions import (
    williamson,
    blochmessiah,
    is_symplectic,
    symplectic_eigenvals,
    sympmat,
)

import strawberryfields as sf
import allfock
import qubovert as qv
import itertools


class BosonicPoly(qv.utils.DictArithmetic):
    def __setitem__(self, key, value):
        if not key or "dag" not in key[-1]:
            return super().__setitem__(key, value)
        if all("dag" in x for x in key):
            return super().__setitem__(key, value)

        num = int(key[-1][4:])
        for i in range(len(key) - 2, -1, -1):
            t = key[i]
            if "dag" not in t and int(t[1:]) == num:
                self[key[:i] + key[i + 1 : -1]] += value

        self[key[-1:] + key[:-1]] += value

    @property
    def offset(self):
        return self[()]


def create_symplectic_from_unitary(u):
    n = len(u)
    O = np.zeros((2 * n, 2 * n))
    O[:n, :n] = u.real
    O[n:, n:] = u.real
    O[:n, n:] = -u.imag
    O[n:, :n] = u.imag
    return O


random_unitary = lambda n: unitary_group.rvs(n)
random_ortho = lambda n: ortho_group.rvs(n)


def random_symplectic(s, real=False):
    n = len(s)
    D = np.diag(np.exp(np.concatenate((s, -s))))
    if not real:
        if n == 1:
            u = np.exp(1j * np.random.random((1, 1)) * 2 * np.pi)
            w = np.exp(1j * np.random.random((1, 1)) * 2 * np.pi)
        else:
            u, w = random_unitary(n), random_unitary(n)
    else:
        if n == 1:
            u = np.random.randint(0, 2, size=(1, 1)).astype(np.float64)
            w = np.random.randint(0, 2, size=(1, 1)).astype(np.float64)
        else:
            u, w = random_ortho(n), random_ortho(n)
    O = create_symplectic_from_unitary(u)
    Op = create_symplectic_from_unitary(w)

    return O @ D @ Op


def overlap(f, S, g, cutoff_dim=10):
    # return |<f| rho(S) |g>|^2

    f = tuple(abs(round(x)) for x in f)
    g = tuple(abs(round(x)) for x in g)

    n = len(f)

    if np.allclose(S.T @ S, np.eye(2 * n)):
        U = S[:n, :n] + 1j * S[n:, :n]
        return allfock.overlap(U, f, g) ** 2
    else:
        print("Using strawberryfields")

    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
    prog = sf.Program(n)

    with prog.context as q:
        for i in range(n):
            sf.ops.Fock(g[i]) | q[i]
        sf.ops.GaussianTransform(S) | q

        result = eng.run(prog)
        state = result.state
        # print(state.all_fock_probs())
        overlap = state.fock_prob(f)

        return overlap


def find_symplectic(Lambda1, Lambda2):

    n = len(Lambda1) // 2

    # print(np.linalg.eigvals((Lambda2.real + Lambda2.real.T) / 2.0))
    # assert 0
    add = 10000 * np.eye(4 * n**2)
    L = (Lambda2.real + Lambda2.real.T) / 2.0 + add
    P, R = williamson(L)

    R = R.reshape((2 * n, 2 * n, 2 * n, 2 * n))
    num = np.sum(R[:, i, :, i] for i in range(2 * n))
    R = R.reshape((4 * n**2, 4 * n**2))
    den = np.trace(R)
    S = num / den ** (1 / 2.0)
    print("R")
    print((R @ P @ R.T - L).round(10))
    print(is_symplectic(S))
    print("S")
    print((np.kron(S, S) @ P @ np.kron(S.T, S.T) - L).round(10))

    assert 0


def rrrr_exp_val(fock, a, b, c, d, e, f, g, h):
    # compute <fock| r[a+ne] r[b+nf] r[c+ng] r[d+nh] |fock>
    pass


def initial_Lambdas(fock):
    Lambda1 = np.diag(np.concatenate((fock, fock)) + 1 / 2).astype(np.complex128)
    n = len(fock)
    for i in range(n):
        Lambda1[i, n + i] = 1j / 2.0
        Lambda1[n + i, i] = -1j / 2.0

    Lambda2 = np.zeros((2 * n, 2 * n, 2 * n, 2 * n)).astype(np.complex128)

    a = [BosonicPoly.create_var(f"a{i}") for i in range(n)]
    adag = [BosonicPoly.create_var(f"adag{i}") for i in range(n)]
    # x = [(a[i] + adag[i]) / np.sqrt(2) for i in range(n)]
    # p = [1j * (adag[i] - a[i]) / np.sqrt(2) for i in range(n)]
    # r = x + p
    initial_poly = BosonicPoly() + np.sqrt(np.prod(scipy.special.factorial(fock)))
    final_poly = BosonicPoly() + np.sqrt(np.prod(scipy.special.factorial(fock)))
    for i in range(n):
        if fock[i]:
            initial_poly *= adag[i] ** int(fock[i])
            final_poly *= a[i] ** int(fock[i])

    # for i in range(2 * n):
    #     for j in range(2 * n):
    #         for k in range(2 * n):
    #             for l in range(2 * n):
    #                 poly = final_poly * r[i] * r[j] * r[k] * r[l] * initial_poly
    #                 Lambda2[i, j, k, l] = poly.offset

    # save time with things that are all zero
    same = lambda x, y: (x[0] == y[0] and x[1] == y[1]) or (
        x[0] == y[1] and x[1] == y[0]
    )
    for i in range(n):
        for j in range(n):
            for w, b, c, d in ((i, i, j, j), (i, j, i, j), (i, j, j, i)):
                for e, f, g, h in itertools.product(*[(0, 1)] * 4):
                    # poly = (
                    #     final_poly
                    #     * r[w + n * e]
                    #     * r[b + n * f]
                    #     * r[c + n * g]
                    #     * r[d + n * h]
                    #     * initial_poly
                    # )
                    # Lambda2[w + n * e, b + n * f, c + n * g, d + n * h] = poly.offset

                    index = w + n * e, b + n * f, c + n * g, d + n * h
                    if Lambda2[index]:
                        continue

                    prefactor = (1j) ** (e + f + g + h) / 4.0
                    Lambda2[index] = 0
                    if same((w, b), (c, d)):
                        Lambda2[index] += (
                            prefactor
                            * (-1) ** (g + h)
                            * (
                                final_poly
                                * adag[w]
                                * adag[b]
                                * a[c]
                                * a[d]
                                * initial_poly
                            ).offset
                        )
                        Lambda2[index] += (
                            prefactor
                            * (-1) ** (e + f)
                            * (
                                final_poly
                                * a[w]
                                * a[b]
                                * adag[c]
                                * adag[d]
                                * initial_poly
                            ).offset
                        )
                    if same((w, c), (b, d)):
                        Lambda2[index] += (
                            prefactor
                            * (-1) ** (f + h)
                            * (
                                final_poly
                                * adag[w]
                                * a[b]
                                * adag[c]
                                * a[d]
                                * initial_poly
                            ).offset
                        )
                        Lambda2[index] += (
                            prefactor
                            * (-1) ** (e + g)
                            * (
                                final_poly
                                * a[w]
                                * adag[b]
                                * a[c]
                                * adag[d]
                                * initial_poly
                            ).offset
                        )
                    if same((w, d), (b, c)):
                        Lambda2[index] += (
                            prefactor
                            * (-1) ** (f + g)
                            * (
                                final_poly
                                * adag[w]
                                * a[b]
                                * a[c]
                                * adag[d]
                                * initial_poly
                            ).offset
                        )
                        Lambda2[index] += (
                            prefactor
                            * (-1) ** (e + h)
                            * (
                                final_poly
                                * a[w]
                                * adag[b]
                                * adag[c]
                                * a[d]
                                * initial_poly
                            ).offset
                        )

    return Lambda1, Lambda2.reshape((4 * n**2, 4 * n**2))


def create_Lambdas(S, fock, noise):
    s1, s2 = initial_Lambdas(fock)
    Lambda1 = S @ s1 @ S.T
    Lambda2 = np.kron(S, S) @ s2 @ np.kron(S.T, S.T)
    if noise:
        Lambda1 += (2 * np.random.random((2 * n, 2 * n)) - 1.0) * noise
        Lambda2 += (2 * np.random.random((4 * n**2, 4 * n**2)) - 1.0) * noise
    return Lambda1, Lambda2


def check(S, fock, noise=0):
    # if noise = 0, then we assume that we can measure Lambda perfectly
    # if noise > 0, then we add noise to our measurements
    Lambda1, Lambda2 = create_Lambdas(S, fock, noise)
    Q, fock1 = find_symplectic(Lambda1, Lambda2)
    return overlap(fock, np.linalg.inv(S) @ Q, fock1, max(10, sum(fock) + sum(fock1)))


def errors(squeeze, fock, noise=0, iters=10, real=False):
    # return the differences between 1 and
    # the magnitude of the overlap that we achieve
    res = [check(random_symplectic(squeeze, real), fock, noise) for _ in range(iters)]
    return [float(1 - x) for x in res]


def test(squeeze, fock, noise=0, iters=10, real=False):
    # return the maximum difference between 1 and
    # the magnitude of the overlap that we achieve
    return max(errors(squeeze, fock, noise, iters, real))


def errors_random(n, noise=0, iters=10, real=False):
    res = []
    for _ in range(iters):
        squeeze = 2 * np.random.random(n) - 1
        fock = [random.randint(0, 3) for _ in range(n)]
        res.extend(errors(squeeze, fock, noise, 1, real))
    return res


def test_random(n, noise=0, iters=10, real=False):
    return max(errors_random(n, noise, iters, real))


if __name__ == "__main__":

    random.seed(0)
    np.random.seed(0)

    n = 2
    test_random(n, 0, 1, True)
