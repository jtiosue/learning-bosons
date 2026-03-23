import numpy as np
from scipy.stats import unitary_group


def random_unitary(n):
    return (
        unitary_group.rvs(n)
        if n > 1
        else np.array([[np.exp(1j * 2 * np.pi * np.random.random())]])
    )


def create_symplectic_from_unitary(u):
    return np.block([[u.real, -u.imag], [u.imag, u.real]])


def q_from_r_unitary(n):
    i = np.eye(n)
    return np.block([[i, 1.0j * i], [i, -1j * i]]) / np.sqrt(2)


def sigma_from_Lambda(Lambda1, Lambda2):
    """
    Create sigma^{(1)}, sigma^{(2)} from Lambda^{(1)}, Lambda^{(2)}
    """
    n = len(Lambda1) // 2
    u = q_from_r_unitary(n)
    uu = np.kron(u, u)
    L1 = u @ Lambda1 @ u.T
    L2 = uu @ Lambda2 @ uu.T
    # this gives us the Lambdas in the q basis.
    # Now we need to take the relevant submatrix
    sigma1, sigma2 = L1[:n, n:], L2.reshape((2 * n,) * 4)[:n, :n, n:, n:].reshape(
        (n**2, n**2)
    )

    # To do: figure out if this should be .conj or not
    return sigma1, sigma2


def fockstate_Lambda_from_sigma(sigma1, sigma2):
    n = len(sigma1)
    sigma2 = sigma2.reshape((n, n, n, n))
    id = np.eye(n)
    # L1, L2 and Lambda1, Lambda2 in the q representation
    L1, L2 = np.zeros((2 * n, 2 * n)), np.zeros((2 * n, 2 * n, 2 * n, 2 * n))
    # for a Fock state, we know that the only nonzero elements of L1 are:
    ## <a adag>
    L1[:n, n:] = sigma1.copy()
    ## <adag a> = -d01 + <a1 adag0>
    L1[n:, :n] = -id + L1[:n, n:].T

    # for a Fock state, we know that the only nonzero elements of L2 are:
    ## a a adag adag
    L2[:n, :n, n:, n:] = sigma2.copy()

    # a adag a adag = a0 (-d12 + a2 adag1) adag3
    ## = -a0 adag3 d12 + a0 a2 adag1 adag3
    L2[:n, n:, :n, n:] = -(L1[:n, n:, None, None] * id[None, None, :, :]).transpose(
        (0, 3, 1, 2)
    ) + L2[:n, :n, n:, n:].transpose((0, 2, 1, 3))

    # a adag adag a = a0 adag1 (-d23 + a3 adag2)
    L2[:n, n:, n:, :n] = -(L1[:n, n:, None, None] * id[None, None, :, :]) + L2[
        :n, n:, :n, n:
    ].transpose((0, 1, 3, 2))

    # <adag a a adag> = <a3 adag2 adag1 a0>.conj()
    L2[n:, :n, :n, n:] = L2[:n, n:, n:, :n].conj().transpose((3, 2, 1, 0))

    # adag a adag a = adag0 (d12 + adag2 a1) a3
    L2[n:, :n, n:, :n] = (L1[n:, :n, None, None] * id[None, None, :, :]).transpose(
        (0, 3, 1, 2)
    ) + L2[n:, n:, :n, :n].transpose((0, 2, 1, 3))

    ## adag0 adag1 a2 a3 = - a2 adag1 d30 - a2 adag0 d13 - adag1 a3 d02 - adag0 a3 d21 + a2 a3 adag0 adag1
    L2[n:, n:, :n, :n] = (
        -(L1[:n, n:, None, None] * id[None, None, :, :]).transpose((2, 1, 3, 0))
        - (L1[:n, n:, None, None] * id[None, None, :, :]).transpose((2, 0, 1, 3))
        - (L1[n:, :n, None, None] * id[None, None, :, :]).transpose((1, 3, 0, 2))
        - (L1[n:, :n, None, None] * id[None, None, :, :]).transpose((0, 3, 2, 1))
        + L2[:n, :n, n:, n:].transpose((2, 3, 0, 1))
    )

    u = q_from_r_unitary(n).conj().T
    uu = np.kron(u, u)
    return u @ L1 @ u.T, uu @ L2.reshape((4 * n**2, 4 * n**2)) @ uu.T


def fockstate_sigma(f):
    n = len(f)
    sigma1 = np.diag([1 + x for x in f])
    sigma2 = np.zeros((n, n, n, n))
    for i in range(n):
        # <f| ai ai aidag aidag |f> = (fi+1)(fi+2)
        sigma2[i, i, i, i] = (f[i] + 1) * (f[i] + 2)
        for j in range(i + 1, n):
            # <f| ai aj aidag ajdag |f> = (fi+1)(fj+1) = <f| ai aj ajdag aidag |f> = ...
            sigma2[i, j, i, j] = (f[i] + 1) * (f[j] + 1)
            sigma2[i, j, j, i] = sigma2[i, j, i, j]
            sigma2[j, i, i, j] = sigma2[i, j, i, j]
            sigma2[j, i, j, i] = sigma2[i, j, i, j]

    return sigma1, sigma2.reshape((n**2, n**2))


def fockstate_Lambda(f):
    return fockstate_Lambda_from_sigma(*fockstate_sigma(f))


def estimate_Lambda_from_samples(samples):

    # might need some factors of pi coming from the heterodyne povm
    nsamples, n = samples.shape
    sconj = samples.conj()
    id = np.eye(n)

    # We first estimate in the q basis. Then we apply the q_from_r_unitary
    # to go to the r basis
    L1 = np.zeros((2 * n, 2 * n), dtype=np.complex128)
    L2 = np.zeros((2 * n,) * 4, dtype=np.complex128)
    # So we have that L1[i, j] = <qi qj>, where q = (a, adag)
    # need to anti normal order (see README). So when i,j<n, i,j>n, or i<n j>n, we're good.
    L1[:n, :n] = samples.T @ samples
    L1[n:, n:] = L1[:n, :n].conj().T  # == np.einsum("ij,ik", sconj, sconj) ?
    L1[:n, n:] = samples.T @ sconj
    L1 /= nsamples
    # when i > n, j<n, we have <a0dag a1> = <-delta + a1 a0dag>
    L1[n:, :n] = -id + L1[:n, n:].T

    # Now L2. there's 16 cases
    # could use Eq 4 of quant-ph/0505180, but since we only care about a few of them,
    # the code will be must faster if we just hard code it.

    # combine = lambda A: np.einsum("ij,kl", A, id)

    # a a a a.
    L2[:n, :n, :n, :n] = (
        np.einsum("ij,ik,il,im", samples, samples, samples, samples) / nsamples
    )

    # adag adag adag adag.
    # L2[n:, n:, n:, n:] = np.einsum("ij,ik,il,im", sconj, sconj, sconj, sconj) / nsamples
    L2[n:, n:, n:, n:] = L2[:n, :n, :n, :n].conj()

    # a a adag adag
    L2[:n, :n, n:, n:] = (
        np.einsum("ij,ik,il,im", samples, samples, sconj, sconj)
        / nsamples
        # np.einsum("mi,mj,mk,ml->ijkl", samples, samples, sconj, sconj) / nsamples
    )

    # a a a adag
    L2[:n, :n, :n, n:] = (
        np.einsum("ij,ik,il,im", samples, samples, samples, sconj) / nsamples
    )

    # <a adag adag adag>
    L2[:n, n:, n:, n:] = (
        np.einsum("ij,ik,il,im", samples, sconj, sconj, sconj) / nsamples
    )

    # adag a a a = -d01 a2a3 +a1 a0dag a2 a3
    ## = -d01 a2a3 - d02 a1 a3 + a1 a2 a0dag a3
    ## = -d01 a2a3 - d02 a1 a3 - d03 a1 a2 + a1 a2 a3 a0dag
    L2[n:, :n, :n, :n] = (
        -(L1[:n, :n, None, None] * id[None, None, :, :]).transpose((2, 3, 0, 1))
        - (L1[:n, :n, None, None] * id[None, None, :, :]).transpose((1, 3, 0, 2))
        - (L1[:n, :n, None, None] * id[None, None, :, :]).transpose((1, 2, 0, 3))
        + L2[:n, :n, :n, n:].transpose((1, 2, 3, 0))
    )

    # adag adag adag a = (adag3 a2 a1 a0).conj()
    L2[n:, n:, n:, :n] = L2[n:, :n, :n, :n].conj().transpose((3, 2, 1, 0))

    # a a a adag  = aa delta23 + aa adag3 a2
    ## = aa delta23 + a0 a2 d13 + a0 adag3 a1 a2
    ## = a0 a1 delta23 + a0 a2 d13 + a1 a2 d03 + adag3 a0 a1 a2
    L2[:n, :n, :n, n:] = (
        L1[:n, :n, None, None] * id[None, None, :, :]
        + (L1[:n, :n, None, None] * id[None, None, :, :]).transpose((0, 2, 1, 3))
        + (L1[:n, :n, None, None] * id[None, None, :, :]).transpose((1, 2, 0, 3))
        + L2[n:, :n, :n, :n].transpose((3, 0, 1, 2))
    )

    ## adag0 adag1 a2 a3 = - a2 adag1 d30 - a2 adag0 d13 - adag1 a3 d02 - adag0 a3 d21 + a2 a3 adag0 adag1
    L2[n:, n:, :n, :n] = (
        -(L1[:n, n:, None, None] * id[None, None, :, :]).transpose((2, 1, 3, 0))
        - (L1[:n, n:, None, None] * id[None, None, :, :]).transpose((2, 0, 1, 3))
        - (L1[n:, :n, None, None] * id[None, None, :, :]).transpose((1, 3, 0, 2))
        - (L1[n:, :n, None, None] * id[None, None, :, :]).transpose((0, 3, 2, 1))
        + L2[:n, :n, n:, n:].transpose((2, 3, 0, 1))
    )

    # a adag a adag = a0 (-d12 + a2 adag1) adag3
    ## = -a0 adag3 d12 + a0 a2 adag1 adag3
    L2[:n, n:, :n, n:] = -(L1[:n, n:, None, None] * id[None, None, :, :]).transpose(
        (0, 3, 1, 2)
    ) + L2[:n, :n, n:, n:].transpose((0, 2, 1, 3))

    # a adag adag a = a0 adag1 (-d23 + a3 adag2)
    L2[:n, n:, n:, :n] = -(L1[:n, n:, None, None] * id[None, None, :, :]) + L2[
        :n, n:, :n, n:
    ].transpose((0, 1, 3, 2))

    # <adag a a adag> = <a3 adag2 adag1 a0>.conj()
    L2[n:, :n, :n, n:] = L2[:n, n:, n:, :n].conj().transpose((3, 2, 1, 0))

    # adag a adag a = adag0 (d12 + adag2 a1) a3
    L2[n:, :n, n:, :n] = (L1[n:, :n, None, None] * id[None, None, :, :]).transpose(
        (0, 3, 1, 2)
    ) + L2[n:, n:, :n, :n].transpose((0, 2, 1, 3))

    # adag a adag adag = (-d01 + a1 adag0) adag2 adag3
    L2[n:, :n, n:, n:] = -(L1[n:, n:, None, None] * id[None, None, :, :]).transpose(
        (2, 3, 0, 1)
    ) + L2[:n, n:, n:, n:].transpose((1, 0, 2, 3))

    # <a a adag a> = <adag3 a2 adag1 adag0>.conj()
    L2[:n, :n, n:, :n] = L2[n:, :n, n:, n:].conj().transpose((3, 2, 1, 0))

    # adag adag a adag = adag adag delta23 + adag adag adag3 a2
    L2[n:, n:, :n, n:] = L1[n:, n:, None, None] * id[None, None, :, :] + L2[
        n:, n:, n:, :n
    ].transpose((0, 1, 3, 2))

    # <a adag a a> = <adag3 adag2 a1 adag0>.conj()
    L2[:n, n:, :n, :n] = L2[n:, n:, :n, n:].conj().transpose((3, 2, 1, 0))

    ## now let's rotate to the r basis
    u = q_from_r_unitary(n).conj().T
    uu = np.kron(u, u)
    L1, L2 = u @ L1 @ u.T, uu @ L2.reshape((4 * n**2,) * 2) @ uu.T
    return L1, L2


# def sigma_from_Lambda(Lambda1, Lambda2):
#     """
#     Create sigma^{(1)}, sigma^{(2)} from Lambda^{(1)}, Lambda^{(2)}
#     """
#     n = len(Lambda1) // 2
#     Lambda2 = Lambda2.reshape((2 * n, 2 * n, 2 * n, 2 * n))
#     sigma1 = np.zeros((n, n), dtype=np.complex128)
#     sigma2 = np.zeros((n, n, n, n), dtype=np.complex128)
#     for i in range(n):
#         for j in range(n):
#             sigma1[i, j] += (1 / 2) * (
#                 Lambda1[i, j]
#                 + Lambda1[n + i, n + j]
#                 + 1j * Lambda1[n + i, j]
#                 - 1j * Lambda1[i, n + j]
#             )
#             for k in range(n):
#                 for l in range(n):
#                     sigma2[i, j, k, l] += (1 / 4) * sum(
#                         (1j) ** (a[0] + a[1])
#                         * (-1j) ** (a[2] + a[3])
#                         * Lambda2[
#                             i + n * a[0], j + n * a[1], k + n * a[2], l + n * a[3]
#                         ]
#                         for a in itertools.product(*[(0, 1)] * 4)
#                     )

#             # this is if we did + cc
#             # sigma1[i, j] = (1 / 2) * (
#             #     Lambda1[i, j] + Lambda1[n + i, n + j] + int(i == j)
#             # )
#             # for k in range(n):
#             #     for l in range(n):
#             #         sigma2[i, j, k, l] += (1 / 4) * (
#             #             Lambda2[i, j, k, l]
#             #             + Lambda2[i + n, j + n, k + n, l + n]
#             #             - Lambda2[i, j, k + n, l + n]
#             #             - Lambda2[i, j + n, k, l + n]
#             #             - Lambda2[i, j + n, k + n, l]
#             #             - Lambda2[i + n, j, k, l + n]
#             #             - Lambda2[i + n, j, k + n, l]
#             #             - Lambda2[i + n, j + n, k, l]
#             #         )

#     sigma2 = sigma2.reshape((n**2, n**2))
#     return sigma1, sigma2


# from thewalrus import perm
# from math import factorial
# def passive_overlap(fock1, W, fock2):
#     """
#     For a passive Gaussian unitary \mathcal{U}_W specified by the n by n
#     unitary matrix W, compute the overlap |<fock1| \mathcal{U}_W |fock2>|,
#     where fock1 and fock2 are length n integer vectors specifying Fock states.
#     """
#     if sum(fock1) != sum(fock2):
#         return 0.0
#     elif len(fock1) == 1:
#         return float(fock1[0] == fock2[0])
#     elif all(x == 0 for x in fock1) and all(y == 0 for y in fock2):
#         return 1.0
#     # N = sum(fock1)
#     rows = []
#     for i, f in enumerate(fock1):
#         for _ in range(f):
#             rows.append(W[i, :])
#     rows = np.array(rows)
#     M = []
#     for i, f in enumerate(fock2):
#         for _ in range(f):
#             M.append(rows[:, i])
#     M = np.array(M).T
#     return np.abs(perm(M)) / np.sqrt(
#         np.prod([factorial(f) for f in fock1]) * np.prod([factorial(f) for f in fock2])
#     )


# def Gaussian_overlap(f, S, g, cutoff_dim_factor=2):
#     """
#     Computes |<f| \mathcal{U}_S |g>| for Fock states f and g and for a
#     symplectic matrix S.

#     cutoff_dim_factor refers to how high in Fock space we truncate.
#     We truncate at sum(f) * cutoff_dim_factor. If S is passive, then
#     cutoff_dim_factor >= 1 will result in no errors.
#     If S is not passive, then no matter what we set cutoff_dim_factor to be,
#     there will be some errors. The larger the squeezing in S, the larger
#     cutoff_dim_factor needs to be set in order to achieve good accuracy.
#     """
#     import strawberryfields as sf
#     f = tuple(abs(round(x)) for x in f)
#     g = tuple(abs(round(x)) for x in g)

#     n = len(f)

#     if np.allclose(S.T @ S, np.eye(2 * n)):
#         U = S[:n, :n] + 1j * S[n:, :n]
#         return passive_overlap(U, f, g)

#     eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim_factor * sum(f)})
#     prog = sf.Program(n)

#     with prog.context as q:
#         for i in range(n):
#             sf.ops.Fock(g[i]) | q[i]
#         sf.ops.GaussianTransform(S) | q

#         result = eng.run(prog)
#         state = result.state
#         overlap = state.fock_prob(f)

#         return np.sqrt(overlap)
