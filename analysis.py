import numpy as np
import matplotlib.pyplot as plt
from thewalrus.random import random_symplectic
from thewalrus import permanent_repeated, perm
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
    heterodyne_samples_from_vacuum,
)


def analyze_algorithm_1(b, ns, nsampless, iters=10, filename="sim"):
    data = np.zeros((len(ns), len(nsampless), iters))
    for i, n in enumerate(ns):
        print(f"Starting n={n}")
        for k in range(iters):
            W = random_unitary(n)
            S = create_symplectic_from_unitary(W)
            # if b == 0:
            #     samples = heterodyne_samples_from_vacuum(n, nsampless[-1])
            # else:
            samples = heterodyne_samples_from_Gaussian_Fock([b] * n, S, nsampless[-1])
            # np.random.shuffle(samples)

            # _, sigma2 = sigma_from_Lambda(*estimate_Lambda_from_samples(samples))
            # print(
            #     np.max(
            #         np.abs(
            #             sigma2
            #             - np.kron(W, W)
            #             @ fockstate_sigma([b] * n)[1]
            #             @ np.kron(W, W).conj().T
            #         )
            #     )
            # )
            # print(
            #     np.max(
            #         np.abs(
            #             np.einsum(
            #                 "ij,ik,il,im",
            #                 samples,
            #                 samples,
            #                 samples.conj(),
            #                 samples.conj(),
            #             )
            #             .conj()
            #             .reshape((n**2, n**2))
            #             / len(samples)
            #             - np.kron(W, W)
            #             @ fockstate_sigma([b] * n)[1]
            #             @ np.kron(W, W).conj().T
            #         )
            #     )
            # )
            # print()
            # # sigma2 = (
            # #     sigma2.reshape((n, n, n, n))
            # #     .transpose((3, 2, 1, 0))
            # #     .reshape((n**2, n**2))
            # # )
            # sigma2 = sigma2.conj()
            # print(
            #     np.max(
            #         np.abs(
            #             sigma2
            #             - np.kron(W, W)
            #             @ fockstate_sigma([b] * n)[1]
            #             @ np.kron(W, W).conj().T
            #         )
            #     )
            # )
            # print()
            # assert 0

            for j, nsamples in enumerate(nsampless):
                np.random.shuffle(samples)
                _, sigma2 = sigma_from_Lambda(
                    *estimate_Lambda_from_samples(samples[:nsamples])
                )

                if "conj" in filename:
                    V = findV(sigma2.conj(), b)
                else:
                    V = findV(sigma2, b)

                # both of these are the same, but the second is probably faster.
                data[i, j, k] = passive_overlap([b] * n, W.conj().T @ V, [b] * n)
                # data[i, j, k] = abs(permanent_repeated(W.conj().T @ V, [b] * n))
                # for b = 1, we could do:
                # if b == 1: data[i, j, k] = abs(perm(W.conj().T @ V))

    with open(f"data/{filename}.txt", "w") as f:
        print(str(data.tolist()), file=f)

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
    start, end = 1e2, 4e6
    analyze_algorithm_1(
        # 1, np.arange(2, 4), np.geomspace(1000, 5000, 5).astype(int), iters=20
        1,
        [2, 3],
        np.geomspace(start, end, 10).astype(int),
        iters=20,
        filename="sim",
    )
    analyze_algorithm_1(
        # 1, np.arange(2, 4), np.geomspace(1000, 5000, 5).astype(int), iters=20
        1,
        [2, 3],
        np.geomspace(start, end, 10).astype(int),
        iters=20,
        filename="simconj",
    )

    # analyze_algorithm_1_noheterodyne(1, np.arange(1, 10), np.geomspace(0.001, 1, 5))
