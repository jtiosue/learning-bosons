import qubovert as qv
import numpy as np
import scipy
from scipy.stats import unitary_group
from collections import defaultdict
from sympy.utilities.iterables import multiset_permutations

factorial = scipy.special.factorial
binomial = scipy.special.comb


def kron(*args):
    if len(args) == 1:
        return args[0]
    return np.kron(kron(*args[:-1]), args[-1])


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

    @property
    def dag(self):
        p = BosonicPoly()
        for key, value in self.items():
            newkey = tuple(
                x.replace("a", "adag", 1).replace("adagdag", "a", 1)
                for x in reversed(key)
            )
            p[newkey] += (value + 0.0j).conjugate()
        return p


def sigmas(fs, gs=None):
    n = len(list(fs.keys())[0])
    a = [BosonicPoly.create_var(f"a{i}") for i in range(n)]
    adag = [BosonicPoly.create_var(f"adag{i}") for i in range(n)]

    gs = gs or fs.copy()

    polyr = BosonicPoly()
    for g, c in gs.items():
        newpoly = BosonicPoly() + c
        for i in range(n):
            newpoly *= adag[i] ** g[i] / np.sqrt(factorial(g[i])) if g[i] else 1
        polyr += newpoly

    polyl = BosonicPoly()
    for f, c in fs.items():
        newpoly = BosonicPoly() + c.conjugate()
        for i in range(n):
            newpoly *= a[i] ** f[i] / np.sqrt(factorial(f[i])) if f[i] else 1
        polyl += newpoly

    sigma1, sigma2 = np.zeros((n, n)), np.zeros((n, n, n, n))

    for i in range(n):
        for j in range(n):
            sigma1[i, j] = (polyl * (a[i] * adag[j]) * polyr).offset
            # sigma1[i, j] = (polyl * (adag[i] * a[j]) * polyr).offset
            for k in range(n):
                for l in range(n):
                    sigma2[i, j, k, l] = (
                        polyl * (a[i] * a[j] * adag[k] * adag[l]) * polyr
                    ).offset
                    # sigma2[i, j, k, l] = (
                    #     polyl * (adag[i] * adag[j] * a[k] * a[l]) * polyr
                    # ).offset

    return sigma1, sigma2.reshape((n**2, n**2))


def sigmas_3(fs, gs=None):
    n = len(list(fs.keys())[0])
    a = [BosonicPoly.create_var(f"a{i}") for i in range(n)]
    adag = [BosonicPoly.create_var(f"adag{i}") for i in range(n)]

    gs = gs or fs.copy()

    polyr = BosonicPoly()
    for g, c in gs.items():
        newpoly = BosonicPoly() + c
        for i in range(n):
            newpoly *= adag[i] ** g[i] / np.sqrt(factorial(g[i])) if g[i] else 1
        polyr += newpoly

    polyl = BosonicPoly()
    for f, c in fs.items():
        newpoly = BosonicPoly() + c.conjugate()
        for i in range(n):
            newpoly *= a[i] ** f[i] / np.sqrt(factorial(f[i])) if f[i] else 1
        polyl += newpoly

    sigma1, sigma2, sigma3 = (
        np.zeros((n, n)),
        np.zeros((n, n, n, n)),
        np.zeros((n, n, n, n, n, n)),
    )

    for i in range(n):
        for j in range(n):
            sigma1[i, j] = (polyl * (a[i] * adag[j]) * polyr).offset
            for k in range(n):
                for l in range(n):
                    sigma2[i, j, k, l] = (
                        polyl * (a[i] * a[j] * adag[k] * adag[l]) * polyr
                    ).offset
                    for m in range(n):
                        for o in range(n):
                            sigma3[i, j, k, l, m, o] = (
                                polyl
                                * (a[i] * a[j] * a[k] * adag[l] * adag[m] * adag[o])
                                * polyr
                            ).offset

    return sigma1, sigma2.reshape((n**2, n**2)), sigma3.reshape((n**3, n**3))


# n = 3
# t = 2
# print([tuple(1 + 2 * t * ((k + j) % n) for k in range(n)) for j in range(n)])
# assert 0


fs = {(0, 5): 1 / np.sqrt(2), (5, 0): 1 / np.sqrt(2)}
# fs = {(1, 5): 1 / np.sqrt(2), (5, 1): 1 / np.sqrt(2)}
# fs = {(2, 6): 1 / np.sqrt(2), (6, 2): 1 / np.sqrt(2)}
sigma1, sigma2 = sigmas(fs)
print(sigma1)
print()
print(sigma2)
print()

U = unitary_group.rvs(2)
print((np.kron(U, U) @ sigma2 @ np.kron(U.conj().T, U.conj().T)).round(6))


fs = {(0, 4): 1 / np.sqrt(2), (4, 0): 1 / np.sqrt(2)}
# fs = {(1, 1, 5): 1 / np.sqrt(3), (1, 5, 1): 1 / np.sqrt(3), (5, 1, 1): 1 / np.sqrt(3)}
sigma1, sigma2, sigma3 = sigmas_3(fs)
print(sigma1)
print()
print(sigma2)
print()
print(sigma3)
print()

U = unitary_group.rvs(2)
print((kron(U, U, U) @ sigma3 @ kron(U.conj().T, U.conj().T, U.conj().T)).round(6))


###############


def sigmas_psi0(n, t):

    sigma1, sigma2, sigma3 = (
        np.zeros((n, n)),
        np.zeros((n, n, n, n)),
        np.zeros((n, n, n, n, n, n)),
    )

    f = np.array([1 + 2 * k * t for k in range(n)])
    for g in multiset_permutations(f):
        g = np.array(g, dtype=float)
        for i in range(n):
            for j in range(n):
                if i == j:
                    sigma1[i, j] += g[i] + 1
                for k in range(n):
                    for l in range(n):
                        if sorted([i, j]) == sorted([k, l]):
                            unique, counts = np.unique(
                                np.array([i, j]), return_counts=True
                            )
                            d = defaultdict(int, zip(unique, counts))
                            mag = np.array([d[a] for a in range(n)])
                            sigma2[i, j, k, l] += np.prod(
                                [
                                    factorial(mag[a]) * binomial(g[a] + mag[a], g[a])
                                    for a in range(n)
                                ]
                            )
                        for m in range(n):
                            for o in range(n):
                                if sorted([i, j, k]) == sorted([l, m, o]):
                                    unique, counts = np.unique(
                                        np.array([i, j, k]), return_counts=True
                                    )
                                    d = defaultdict(int, zip(unique, counts))
                                    mag = np.array([d[a] for a in range(n)])
                                    sigma3[i, j, k, l, m, o] += np.prod(
                                        [
                                            factorial(mag[a])
                                            * binomial(g[a] + mag[a], g[a])
                                            for a in range(n)
                                        ]
                                    )

    fac = factorial(n)
    return (
        sigma1 / fac,
        sigma2.reshape((n**2, n**2)) / fac,
        sigma3.reshape((n**3, n**3)) / fac,
    )


# sigma1, sigma2, sigma3 = sigmas_psi0(2, 3)
# U = unitary_group.rvs(2)
# print(sigma1)
# print()
# print(sigma2)
# print()
# print((kron(U, U) @ sigma2 @ kron(U.conj().T, U.conj().T)).round(6))
# print()
# print(sigma3)
# print()
# print((kron(U, U, U) @ sigma3 @ kron(U.conj().T, U.conj().T, U.conj().T)).round(6))
