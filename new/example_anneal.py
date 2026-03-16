from __future__ import annotations

import numpy as np

from anneal_wrapper import sample_heterodyne


def main() -> None:
    n = 2
    f = np.array([1, 0], dtype=np.int32)
    U = np.array([[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]], dtype=np.complex128)
    # P = np.array([[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]], dtype=np.complex128)
    # U = np.eye(n, dtype=np.complex128)
    P = np.eye(n, dtype=np.complex128)
    ls = np.array([0.0, 0.0], dtype=np.float64)
    initial_alpha = np.array([0.5 + 0.5j, 0.5 + 0.5j], dtype=np.complex128)

    samples = sample_heterodyne(
        n=n,
        f=f,
        U=U,
        P=P,
        ls=ls,
        nsamples=5000,
        initial_anneal=1000,
        delta=10,
        initial_alpha=initial_alpha,
    )

    print("samples.shape =", samples.shape)
    # print(samples[::100])
    print(samples.conj().T @ samples / len(samples))


if __name__ == "__main__":
    main()