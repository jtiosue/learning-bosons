import numpy as np
from tqdm import tqdm
from math import factorial
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
    fockstate_Lambda_from_sigma,
    create_symplectic_from_unitary,
)
from sample import (
    sample_heterodyne_passive_fock1,
    sample_heterodyne_passive_fock,
    sample_heterodyne_Gaussian_fock,
)
from datetime import datetime


def analyze_algorithm_1_fock1_memorytight(
    ns, nsampless, iters=10, filename="sim", mem_thresh=100_00
):
    """
    Run a full analysis of algorithm 1 for learning the state U|1^n> for arbitrary passive linear optical circuit U.
    This version is memory tight, and will not store all samples in memory at once.
    Instead, it will sample in batches of size mem_thresh, and accumulate the results.
    """
    data = np.zeros((len(ns), len(nsampless), iters))
    for i, n in enumerate(ns):
        print(f"Starting n={n}")
        # for k in tqdm(range(iters)):
        for k in range(iters):
            print(f"Starting iter {k} at", datetime.now())
            W = random_unitary(n)
            for j, nsamples in enumerate(nsampless):
                sigma2 = np.zeros((n, n, n, n), dtype=np.complex128)
                for _ in range(nsamples // mem_thresh):
                    samples = sample_heterodyne_passive_fock1(W, mem_thresh)
                    sconj = samples.conj()
                    sigma2 += np.einsum("ij,ik,il,im", samples, samples, sconj, sconj)

                if nsamples % mem_thresh:
                    samples = sample_heterodyne_passive_fock1(W, nsamples % mem_thresh)
                    sconj = samples.conj()
                    sigma2 += np.einsum("ij,ik,il,im", samples, samples, sconj, sconj)

                sigma2 /= nsamples
                sigma2.resize((n**2, n**2))

                V = findV(sigma2, 1)

                data[i, j, k] = abs(perm(W.conj().T @ V))

        with open(f"data/{filename}.txt", "w") as f:
            print(str(ns.tolist() if isinstance(ns, np.ndarray) else ns), file=f)
            print(str(nsampless.tolist()), file=f)
            print(str(data.tolist()), file=f, end="")

    plot(filename)


def plot(filename, threshold=0.75):

    with open(f"data/{filename}.txt") as f:
        ns = np.array(eval(f.readline()))
        nsampless = np.array(eval(f.readline()))
        data = np.array(eval(f.readline()))

    f = plt.figure()
    for i, n in enumerate(ns):
        # plt.errorbar(
        #     nsampless,
        #     np.mean(data[i], axis=1),
        #     np.std(data[i], axis=1),
        #     label=f"n = {n}",
        #     marker="o",
        # )
        plt.plot(
            nsampless,
            np.mean(data[i], axis=1),
            label=f"n = {n}",
            marker="o",
        )
    plt.xlabel("nsamples")
    plt.ylabel("overlap")
    plt.title(f"U_W |1^n>")
    plt.legend()
    ax = plt.gca()
    ax.set_xscale("log")
    f.savefig(f"data/{filename}.pdf")
    plt.close()

    plotns, num_samples, eb = [], [], []
    f = plt.figure()
    for i, n in enumerate(ns):
        d = np.argmax(data[i] >= threshold, axis=0)
        d = d[d > 0]
        if len(d):
            plotns.append(n)
            num_samples.append(np.mean(nsampless[d]))
            eb.append(np.std(nsampless[d]))
    f = plt.figure()
    # plt.errorbar(plotns, num_samples, eb, marker="o")
    plt.plot(plotns, num_samples, marker="o")
    plt.xlabel("n")
    plt.ylabel("nsamples")
    plt.title(f"U_W |1^n>, num samples to reach overlap = {threshold}")
    ax = plt.gca()
    # ax.set_xscale("log")
    ax.set_yscale("log")
    f.savefig(f"data/{filename}_theshold.pdf")
    plt.close()


def run_algorithm_1(n: int, nsamples: int):
    """
    Learning algorithm 2 for learning the state U|f>
    for arbitrary fock state |f> and passive linear optical circuit U.

    """
    b = np.random.randint(1, 3)
    W = random_unitary(n)
    heterodyne_samples = sample_heterodyne_passive_fock1(W, nsamples)
    Lambda1, Lambda2 = estimate_Lambda_from_samples(heterodyne_samples)
    sigma1, sigma2 = sigma_from_Lambda(Lambda1, Lambda2)
    V = findV(sigma2, b)
    overlap = abs(permanent_repeated(W.conj().T @ V, [b] * n)) / (factorial(b) ** n)
    return overlap


def run_algorithm_2(n: int, nsamples: int):
    """
    Learning algorithm 2 for learning the state U|f>
    for arbitrary fock state |f> and passive linear optical circuit U.

    """
    W = random_unitary(n)
    f = np.random.randint(0, 3, size=n)
    heterodyne_samples = sample_heterodyne_passive_fock(W, f, nsamples)
    Lambda1, Lambda2 = estimate_Lambda_from_samples(heterodyne_samples)
    sigma1, sigma2 = sigma_from_Lambda(Lambda1, Lambda2)
    V = findVFock(sigma1, sigma2)
    overlap = abs(permanent_repeated(W.conj().T @ V, f))
    return overlap


def run_algorithm_3(n: int, nsamples: int):
    """
    Learning algorithm 3 for learning the state S|f>
    for arbitrary fock state |f> and Gaussian unitary specified
    by symplectic matrix S.

    """
    S = random_symplectic(n)
    f = np.random.randint(0, 3, size=n)
    heterodyne_samples = sample_heterodyne_Gaussian_fock(S, f, nsamples)
    Lambda1, Lambda2 = estimate_Lambda_from_samples(heterodyne_samples)
    Q = findQ(Lambda1, Lambda2)

    raise NotImplementedError(
        "Computing overlap for arbitrary Gaussian unitaries is not implemented yet."
    )
    overlap = 1
    return overlap


if __name__ == "__main__":
    # run a single instance of algorithm 1 for
    # n=3 modes and nsamples=10000 hetoeryne samples
    # it will print the fidelity of the learned state with the true state
    # print(run_algorithm_1(n=3, nsamples=10000))

    # run full analysis of algorithm 1
    analyze_algorithm_1_fock1_memorytight(
        ns=[2, 3],
        nsampless=np.geomspace(1e1, 1e5, 20).astype(int),
        iters=50,
        filename="test_fock1_sim",
        mem_thresh=100_000,
    )
