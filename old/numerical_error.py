from bosons import errors
import matplotlib.pyplot as plt
import numpy as np


def run_and_plot(c, alpha, ns, iters, color, axes):
    res, bars, bound = [], [], []

    print("Finished n = ", end="")

    for n in ns:
        noise = (1 / np.sqrt(20)) * n ** (-4 - alpha - 1 / 2)
        # bound.append(c * n ** (-alpha))
        bound.append((c * n ** (-alpha)))  ## require the overlap to be > 1 - tol
        err = errors(n, noise=noise, iters=iters)
        res.append(np.mean(err))
        # bars.append(np.std(err) / np.sqrt(len(err)))
        bars.append([res[-1] - min(err), max(err) - res[-1]])
        print(f"{n}, ", end="", flush=True)
    print()

    bars = np.array(bars).T

    axes.plot(
        ns,
        bound,
        ":",
        color=color,
        label=rf"analytic bound ($c = {c}, \alpha = {alpha}$)",
    )
    axes.errorbar(
        ns,
        res,
        bars,
        color=color,
        label=rf"numerical error ($c = {c}, \alpha = {alpha}$)",
    )


if __name__ == "__main__":

    c = 1
    iters = 100
    ns = range(5, 16)

    f = plt.figure()
    axes = plt.gca()

    for alpha, color in zip((1, 2, 3, 4), ("b", "orange", "r", "k")):
        print("Starting alpha =", alpha)
        run_and_plot(c, alpha, ns, iters, color, axes)

    axes.set_xscale("log")
    axes.set_yscale("log")
    plt.legend()
    plt.xlabel(r"$n$")
    plt.ylabel(
        r"$1 - \left | \langle 1^n\vert \rho(W^\dag V) \vert 1^n\rangle \right |$"
    )
    plt.title("Analytic error bound vs numerics")
    f.set_size_inches(13, 8)
    plt.show()
