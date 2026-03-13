import qubovert as qv
import numpy as np
import scipy


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


n = 8
a = [BosonicPoly.create_var(f"a{i}") for i in range(n)]
adag = [BosonicPoly.create_var(f"adag{i}") for i in range(n)]
import time

t0 = time.time()
print(len(sum((a[i] + a[i + 1]) ** 2 for i in range(n - 1))))
print(time.time() - t0)

assert 0


def exp_value(f, U, s, Up, g):
    n = len(f)
    a = [BosonicPoly.create_var(f"a{i}") for i in range(n)]
    adag = [BosonicPoly.create_var(f"adag{i}") for i in range(n)]
    polyr = BosonicPoly() + 1
    polyl = BosonicPoly() + 1
    for i in range(n):
        for _ in range(g[i]):
            polyr *= qv.utils.sum(Up[i, j].conj() * adag[j] for j in range(n))
        for _ in range(f[i]):
            polyl *= qv.utils.sum(U[j, i].conj() * a[j] for j in range(n))

    newpolyr = BosonicPoly()

    for term, coef in polyr.items():
        newterm = BosonicPoly() + 1
        for x in term:
            dag = "dag" in x
            num = int(x[4:]) if dag else int(x[1:])
            w = adag[num] if dag else a[num]
            y = adag[num] if not dag else a[num]
            newterm *= np.cosh(s[num]) * w + np.sinh(s[num]) * y
        newpolyr += coef * newterm

    return (polyl * newpolyr).offset / np.sqrt(
        np.prod(scipy.special.factorial(f)) * np.prod(scipy.special.factorial(g))
    )


print(exp_value([2, 3], np.eye(2), [0, 1], np.eye(2), [2, 3]))


###


r = [qv.utils.DictArithmetic.create_var(f"x{i}") for i in ("i", "j", "k", "l")]
r += [qv.utils.DictArithmetic.create_var(f"p{i}") for i in ("i", "j", "k", "l")]
term = (1 / 8) * sum(
    (1j) ** (a + b + c + d) * r[0 + 4 * a] * r[1 + 4 * b] * r[2 + 4 * c] * r[3 + 4 * d]
    for a in range(2)
    for b in range(2)
    for c in range(2)
    for d in range(2)
) + (1 / 8) * sum(
    (-1j) ** (a + b + c + d) * r[3 + 4 * a] * r[2 + 4 * b] * r[1 + 4 * c] * r[0 + 4 * d]
    for a in range(2)
    for b in range(2)
    for c in range(2)
    for d in range(2)
)
# print(term)

poly = term * 0
for k, coef in term.items():
    poly[k] = coef.real

print(poly)
print()

poly = term * 0
for k, coef in term.items():
    poly[k] = coef.imag

print(poly)
print()
