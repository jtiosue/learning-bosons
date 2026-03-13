# from symplectic import random_symplectic
import numpy as np
import qubovert as qv
import itertools


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


a = [
    BosonicPoly.create_var("a0"),
    BosonicPoly.create_var("a1"),
]
a0, a1 = a
adag = [
    BosonicPoly.create_var("adag0"),
    BosonicPoly.create_var("adag1"),
]
a0dag, a1dag = adag
# psi0 = 1 / np.sqrt(2) * a1dag * a1dag
# psi0dag = 1 / np.sqrt(2) * a1 * a1
# psi0 = (
#     1
#     / np.sqrt(3)
#     * (1 / np.sqrt(2) * a1dag * a1dag + 1 / np.sqrt(2) * a2dag * a2dag + a1dag * a2dag)
# )
# psi1 = a1dag * a2dag
# psi2 = 1 / 2 * (a1dag * a1dag + a2dag * a2dag)

psi0 = BosonicPoly() + 1
psi1 = a1dag
psi2 = 1 / np.sqrt(2) * a1dag**2 + 1 / np.sqrt(1.4)

psi0 /= np.sqrt((psi0.dag * psi0).offset)
psi1 /= np.sqrt((psi1.dag * psi1).offset)
psi2 /= np.sqrt((psi2.dag * psi2).offset)

psi0dag = psi0.dag
psi1dag = psi1.dag
psi2dag = psi2.dag

np.testing.assert_allclose(
    [(psi0dag * psi0).offset, (psi1dag * psi1).offset, (psi2dag * psi2).offset], 1
)

x = [1 / np.sqrt(2) * (a[i] + adag[i]) for i in range(len(a))]
p = [1.0j / np.sqrt(2) * (adag[i] - a[i]) for i in range(len(a))]
r = x + p

firstMom_0 = np.zeros(len(r), dtype=np.complex128)
firstMom_1 = np.zeros(len(r), dtype=np.complex128)
firstMom_2 = np.zeros(len(r), dtype=np.complex128)

tSigma_0 = np.zeros((len(r),) * 4, dtype=np.complex128)
tSigma_1 = np.zeros((len(r),) * 4, dtype=np.complex128)
tSigma_2 = np.zeros((len(r),) * 4, dtype=np.complex128)

covSigma_0 = np.zeros((len(r),) * 2, dtype=np.complex128)
covSigma_1 = np.zeros((len(r),) * 2, dtype=np.complex128)
covSigma_2 = np.zeros((len(r),) * 2, dtype=np.complex128)

for i in range(len(r)):
    firstMom_0[i] = (psi0dag * r[i] * psi0).offset
    firstMom_1[i] = (psi1dag * r[i] * psi1).offset
    firstMom_2[i] = (psi2dag * r[i] * psi2).offset

for i in range(len(r)):
    for j in range(len(r)):
        op = (r[i] - firstMom_0[i]) * (r[j] - firstMom_0[j])
        covSigma_0[i, j] = (psi0dag * op * psi0).offset
        op = (r[i] - firstMom_1[i]) * (r[j] - firstMom_1[j])
        covSigma_1[i, j] = (psi1dag * op * psi1).offset
        op = (r[i] - firstMom_2[i]) * (r[j] - firstMom_2[j])
        covSigma_2[i, j] = (psi2dag * op * psi2).offset

for i in itertools.product(range(len(r)), repeat=4):
    op = (
        (r[i[0]] - firstMom_0[i[0]])
        * (r[i[1]] - firstMom_0[i[1]])
        * (r[i[2]] - firstMom_0[i[2]])
        * (r[i[3]] - firstMom_0[i[3]])
    )
    tSigma_0[i] = (psi0dag * op * psi0).offset
    op = (
        (r[i[0]] - firstMom_1[i[0]])
        * (r[i[1]] - firstMom_1[i[1]])
        * (r[i[2]] - firstMom_1[i[2]])
        * (r[i[3]] - firstMom_1[i[3]])
    )
    tSigma_1[i] = (psi1dag * op * psi1).offset
    op = (
        (r[i[0]] - firstMom_2[i[0]])
        * (r[i[1]] - firstMom_2[i[1]])
        * (r[i[2]] - firstMom_2[i[2]])
        * (r[i[3]] - firstMom_2[i[3]])
    )
    tSigma_2[i] = (psi2dag * op * psi2).offset

reshape = (len(r) ** 2, len(r) ** 2)
tSigma_0 = tSigma_0.reshape(reshape)
tSigma_1 = tSigma_1.reshape(reshape)
tSigma_2 = tSigma_2.reshape(reshape)

# print(tSigma_0)

n = len(a)
iJ = np.zeros((2 * n, 2 * n), dtype=np.complex128)
for i in range(n):
    iJ[i, n + i] = 1.0j
    iJ[n + i, i] = -1.0j
iJiJ = kron(iJ, iJ)
SWAP = np.zeros(((2 * n),) * 4, dtype=np.complex128)
for i in range(2 * n):
    for j in range(2 * n):
        SWAP[i, j, j, i] += 1
SWAP = SWAP.reshape(((2 * n) ** 2,) * 2)

eig = lambda tSigma: list(
    sorted(
        np.linalg.eigvals(iJiJ @ tSigma).real.round(6).tolist()
        + np.linalg.eigvals(iJiJ @ SWAP @ tSigma).real.round(6).tolist()
    )
)
coveig = lambda Sigma: list(
    sorted(np.linalg.eigvals(iJ @ Sigma).real.round(6).tolist())
)


print("Degree two invariants:\n")
print("|0>")
print(coveig(covSigma_0))
print()
print("|1>")
print(coveig(covSigma_1))
print()

print("|0> + |2>")
print(coveig(covSigma_2))
print()

print("All invariants:\n")

print("|00>")
print(eig(tSigma_0))
print()
print("|1>")
print(eig(tSigma_1))
print()
print("|0> + |2>")
print(eig(tSigma_2))
print()
print(eig(tSigma_1) == eig(tSigma_2))
