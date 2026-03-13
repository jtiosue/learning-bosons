from allfock import errors
import matplotlib.pyplot as plt
import numpy as np
import random


def run_and_plot_vs_n(ns, iters, color, axes):
    res, bars, bound = [], [], []

    print("Finished n = ", end="")
    N = 2
    for n in ns:
        fock = [random.randint(0, N) for _ in range(n)]
        noise = 0.01
        err = errors(n, fock, noise=noise, iters=iters)
        res.append(np.mean(err))
        bars.append(np.std(err) / np.sqrt(len(err)))
        print(f"{n}, ", end="", flush=True)
    print()

    bars = np.array(bars).T

    axes.errorbar(ns, res, bars, color=color, label="vs n")


def run_and_plot_vs_N(Ns, iters, color, axes):
    res, bars, bound = [], [], []

    print("Finished N = ", end="")
    n = 5
    for N in Ns:
        fock = [random.randint(0, N) for _ in range(n)]
        noise = 0.01
        err = errors(n, fock, noise=noise, iters=iters)
        res.append(np.mean(err))
        bars.append(np.std(err) / np.sqrt(len(err)))
        print(f"{N}, ", end="", flush=True)
    print()

    bars = np.array(bars).T

    axes.errorbar(Ns, res, bars, color=color, label="vs N")


if __name__ == "__main__":

    iters = 10
    ns = range(5, 16)

    f = plt.figure()
    axes = plt.gca()

    run_and_plot_vs_n(ns, iters, "r", axes)
    run_and_plot_vs_N(range(1, 8), iters, "b", axes)

    # axes.set_xscale("log")
    axes.set_yscale("log")
    plt.legend()
    plt.xlabel(r"$n,N$")
    plt.ylabel(r"$1 - \left | \langle f\vert \rho(W^\dag V) \vert f \rangle \right |$")
    f.set_size_inches(13, 8)

    plt.show()
