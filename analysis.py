import numpy as np
import matplotlib.pyplot as plt
from thewalrus.random import random_symplectic
from algorithm1 import findV
from algorithm2 import findVFock
from algorithm3 import findQ
from methods import (
    sigma_from_Lambda,
    random_unitary,
    estimate_Lambda_from_samples,
    fockstate_sigma,
    create_symplectic_from_unitary,
)
from heterodyne import (
    passive_overlap,
    Gaussian_overlap,
    heterodyne_samples_from_Gaussian_Fock,
)


def analyze_algorithm_1(b, ns, nsampless, iters=10, filename="sim"):
    data = np.zeros((len(ns), len(nsampless), iters))
    for i, n in enumerate(ns):
        print(f"Starting n={n}")
        for k in range(iters):
            W = random_unitary(n)
            S = create_symplectic_from_unitary(W)
            samples = heterodyne_samples_from_Gaussian_Fock([b] * n, S, nsampless[-1])
            for j, nsamples in enumerate(nsampless):
                _, sigma2 = sigma_from_Lambda(
                    *estimate_Lambda_from_samples(samples[:nsamples])
                )
                V = findV(sigma2, b)
                data[i, j, k] = passive_overlap([b] * n, W.conj().T @ V, [b] * n)

    f = plt.figure()
    for i, n in enumerate(ns):
        plt.errorbar(
            nsampless,
            np.mean(data[i], axis=1),
            np.std(data[i], axis=1),
            label=f"n = {n}",
            marker="o",
        )
    plt.xlabel("nsamples")
    plt.ylabel("overlap")
    plt.title(f"U_W |b^n> with b = {b}")
    plt.legend()
    ax = plt.gca()
    ax.set_xscale("log")
    f.savefig(f"data/{filename}.pdf")
    plt.close()


def analyze_algorithm_1_noheterodyne(
    b, ns, errors, iters=10, filename="sim_noheterodyne"
):
    data = np.zeros((len(ns), len(errors), iters))
    for i, n in enumerate(ns):
        print(f"Starting n={n}")
        _, sigma2_0 = fockstate_sigma([b] * n)
        for k in range(iters):
            W = random_unitary(n)
            WW = np.kron(W, W)
            sigma2 = WW @ sigma2_0 @ WW.conj().T
            for j, error in enumerate(errors):
                # add some noise to it
                sigma2noise = sigma2 + np.random.normal(0, error, size=sigma2.shape)

                V = findV(sigma2noise, b)
                data[i, j, k] = passive_overlap([b] * n, W.conj().T @ V, [b] * n)

    f = plt.figure()
    for i, n in enumerate(ns):
        plt.errorbar(
            errors,
            np.mean(data[i], axis=1),
            np.std(data[i], axis=1),
            label=f"n = {n}",
            marker="o",
        )
    plt.xlabel("errors")
    plt.ylabel("overlap")
    plt.title(f"U_W |b^n> with b = {b}")
    plt.legend()
    ax = plt.gca()
    ax.set_xscale("log")
    f.savefig(f"data/{filename}.pdf")
    plt.close()


if __name__ == "__main__":
    # analyze_algorithm_1(
    #     1, np.arange(2, 3), np.geomspace(500000, 500000, 1).astype(int), iters=2
    # )
    analyze_algorithm_1_noheterodyne(1, np.arange(1, 10), np.geomspace(0.001, 1, 5))
