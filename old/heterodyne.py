import numpy as np
import strawberryfields as sf
import strawberryfields.ops as ops
from methods import q_from_r_unitary


def heterodyne_samples_from_vacuum(n, nsamples):
    # p(alpha) = e^{-|alpha|^2}/pi = |<alpha|0>|^2
    real = np.random.multivariate_normal([0] * n, np.diag([1 / 2] * n), nsamples)
    im = np.random.multivariate_normal([0] * n, np.diag([1 / 2] * n), nsamples)
    return real + 1j * im


def heterodyne_samples_from_passive_Fock(f, W, nsamples):
    """
    Sample heterodyne outcomes from \mathcal{U}_W |f1,...,fn>.
    W should be n by n.

    We return an array of shape (nsamples, n)
    """
    n = len(f)
    eng = sf.Engine("bosonic", backend_options={"cutoff_dim": sum(f) + 1})
    prog = sf.Program(n)

    with prog.context as q:
        for i, fi in enumerate(f):
            ops.Fock(fi) | q[i]
        ops.PassiveChannel(W) | q
        for qm in q:
            ops.MeasureHeterodyne() | qm

    result = eng.run(prog, shots=nsamples)
    # eng.reset()

    return result.samples


def heterodyne_samples_from_passive_Fock(f, W, nsamples):
    n = len(f)
    eng = sf.Engine("fock", backend_options={"cutoff_dim": sum(f) + 1})
    prog = sf.Program(n)

    with prog.context as q:
        for i, fi in enumerate(f):
            ops.Fock(fi) | q[i]
        ops.PassiveChannel(W) | q
        for qm in q:
            ops.MeasureHeterodyne() | qm

    result = np.zeros((nsamples, n), dtype=np.complex128)
    for i in range(nsamples):
        result[i, :] = eng.run(prog).samples
        eng.reset()

    return result


def heterodyne_samples_from_Gaussian(S, nsamples):
    n = len(S) // 2
    eng = sf.Engine("gaussian")
    prog = sf.Program(n)

    with prog.context as q:
        ops.GaussianTransform(S) | q
        for qm in q:
            ops.MeasureHeterodyne() | qm

    result = np.zeros((nsamples, n), dtype=np.complex128)
    for i in range(nsamples):
        result[i, :] = eng.run(prog).samples
        eng.reset()

    return result


# def heterodyne_samples_from_Gaussian_Fock(f, S, nsamples, cutoff_dim_factor=2):
#     """
#     Sample heterodyne outcomes from \mathcal{U}_S |f1,...,fn>.

#     We return an array of shape (nsamples, n)

#     cutoff_dim_factor refers to how high in Fock space we truncate.
#     We truncate at sum(f) * cutoff_dim_factor. If S is passive, then
#     cutoff_dim_factor >= 1 will result in no errors.
#     If S is not passive, then no matter what we set cutoff_dim_factor to be,
#     there will be some errors. The larger the squeezing in S, the larger
#     cutoff_dim_factor needs to be set in order to achieve good accuracy.
#     """
#     n = len(f)
#     eng = sf.Engine(
#         "bosonic", backend_options={"cutoff_dim": cutoff_dim_factor * sum(f)}
#     )
#     prog = sf.Program(n)

#     with prog.context as q:
#         for i, fi in enumerate(f):
#             ops.Fock(fi) | q[i]
#         ops.GaussianTransform(S) | q
#         for qm in q:
#             ops.MeasureHeterodyne() | qm

#     result = eng.run(prog, shots=nsamples)
#     # eng.reset()

#     return result

#########


def heterodyne_samples_from_Gaussian_Fock(f, S, nsamples, cutoff_dim_factor=2):
    f = list(f)
    m = len(f)
    S = np.asarray(S, dtype=float)

    if S.shape != (2 * m, 2 * m):
        raise ValueError(f"S must have shape {(2*m, 2*m)}, got {S.shape}")

    prog = sf.Program(2 * m)

    with prog.context as q:
        # Prepare the input Fock product state |f>
        for i, n in enumerate(f):
            ops.Fock(n) | q[i]

        # Apply the Gaussian unitary on the signal modes
        ops.GaussianTransform(S) | tuple(q[:m])

        # Add m vacuum ancillas implicitly on q[m:2m] and implement heterodyne
        for i in range(m):
            ops.BSgate(np.pi / 4, 0.0) | (q[i], q[m + i])
            ops.MeasureX | q[i]
            ops.MeasureP | q[m + i]

    eng = sf.Engine(
        "fock", backend_options={"cutoff_dim": (sum(f) + 1) * cutoff_dim_factor}
    )
    result = np.zeros((nsamples, 2 * m))
    for i in range(nsamples):
        res = eng.run(prog).samples
        eng.reset()
        result[i, :] = res

    # result.samples has shape (shots, 2m)
    xvals = result[:, :m]
    pvals = result[:, m:]

    # Convert homodyne pair to coherent-state label alpha
    # if sf_hbar2:
    #     alpha = (xvals + 1j * pvals) / 2.0
    # else:
    alpha = (xvals + 1j * pvals) / np.sqrt(2.0)

    return alpha


#######


"""



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
    L2[:n, :n, :n, :n].conj()

    # adag a a a = -d01 a2a3 +a1 a0dag a2 a3
    ## = -d01 a2a3 - d02 a1 a3 + a1 a2 a0dag a3
    ## = -d01 a2a3 - d02 a1 a3 - d03 a1 a2 + a1 a2 a3 a0dag
    L2[n:, :n, :n, :n] = (
        -(L1[:n, :n, None, None] * id[None, None, :, :]).transpose((2, 3, 0, 1))
        - (L1[:n, :n, None, None] * id[None, None, :, :]).transpose((1, 3, 0, 2))
        - (L1[:n, :n, None, None] * id[None, None, :, :]).transpose((1, 2, 0, 3))
        + np.einsum("ij,ik,il,im", sconj, samples, samples, samples) / nsamples
    )

    # adag adag a a = -adag0 a3 d12 + adag0 a2 adag1 a3
    ## = -adag0 a3 d12 -d02 a1dag a3 + a2 adag0 adag1 a3
    ## = -adag0 a3 d12 -d02 a1dag a3 - a2 adag0 d13 + a2 adag0 a3 adag1
    ## = -adag0 a3 d12 -d02 a1dag a3 - a2 adag0 d13 - a2 adag1 d03 + a2 a3 adag0 adag1
    L2[n:, n:, :n, :n] = (
        -(L1[n:, :n, None, None] * id[None, None, :, :]).transpose((0, 3, 1, 2))
        - (L1[n:, :n, None, None] * id[None, None, :, :]).transpose((1, 3, 0, 2))
        - (L1[:n, n:, None, None] * id[None, None, :, :]).transpose((2, 0, 1, 3))
        - (L1[:n, n:, None, None] * id[None, None, :, :]).transpose((2, 1, 0, 3))
        + np.einsum("ij,ik,il,im", sconj, sconj, samples, samples) / nsamples
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

    # <a adag adag adag> = <a3 a2 a1 adag0>.conj()
    L2[:n, n:, n:, n:] = L2[:n, :n, :n, n:].conj().transpose((3, 2, 1, 0))

    # a a adag adag = a0 (d12 + adag2 a1) adag3
    ## = a0 adag3 d12 + a0 adag2 a1 adag3
    ## = a0 adag3 d12 + a0 adag2 d13 + a0 adag2 adag3 a1
    ## = a0 adag3 d12 + a0 adag2 d13 + adag3 a1 d02 + adag2 a0 adag3 a1
    ## = a0 adag3 d12 + a0 adag2 d13 + adag3 a1 d02 + adag2 a1 d03 + adag2 adag3 a0 a1
    L2[:n, :n, n:, n:] = (
        (L1[:n, n:, None, None] * id[None, None, :, :]).transpose((0, 3, 1, 2))
        + (L1[:n, n:, None, None] * id[None, None, :, :]).transpose((0, 2, 1, 3))
        + (L1[n:, :n, None, None] * id[None, None, :, :]).transpose((3, 1, 0, 2))
        + (L1[n:, :n, None, None] * id[None, None, :, :]).transpose((2, 1, 0, 3))
        + L2[n:, n:, :n, :n].transpose((2, 3, 0, 1))
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
    return u @ L1 @ u.T, uu @ L2.reshape((4 * n**2,) * 2) @ uu.T


"""

"""
This was halfway done for homodyne

def estimate_Lambda_from_samples(xsamples, psamples, xpsamples):
    _, n = xsamples.shape
    quadsamples = np.hstack((xsamples, psamples))

    ## first Lambda1. Let's just start off filling up everything. Then we will
    # refill in the problematic ones (where non commutation matters)
    Lambda1 = np.einsum("ij,ik", quadsamples, quadsamples)
    # now the problematic part is expectation value of x_i p_i.
    for i in range(n):
        # <xp> = 1/2 <xp + xp> = 1/2 <xp + xp - px + px> = 1/2 <xp+px> + 1/2[x,p] = = 1/2 <xp+px> + i
        # so we just need to compute 1/2<xp+px>. This is done by using the xp heterodyne that used angle
        # pi/4. That measurement measures in the eigenbasis of z = 1/sqrt2 (x + p).
        # noting that z^2 = (x^2 + p^2)/2 + (xp+px)/2, we can take the expectation value we get from xpsamples
        # and subtract the expectation of x^2 and p^2 that we already computed.
        # namely xp = i + z^2 - x^2+p^2/2
        Lambda1[i, n + i] = (
            1.0j
            + np.dot(xpsamples[:, i], xpsamples[:, i])
            - (Lambda1[i, i] + Lambda1[n + i, n + i]) / 2.0
        )
        # <px> = <px - xp + xp> = <xp> - 2i
        Lambda1[n + i, i] = Lambda1[i, n + i] - 2.0j

    ## now Lambda2. Let's just start off filling up everything. Then we will
    # refill in the problematic ones (where non commutation matters)
    Lambda2 = np.einsum(
        "ij,ik,il,im", quadsamples, quadsamples, quadsamples, quadsamples
    )
    # now we come back to the problematic ones
    # we first deal with when there is one x and one p on the same mode in the observable
    # again we use that z^2 = (x^2 + p^2)/2 + (xp+px)/2.
    # so for the epectation value of xi pi rj rk, where r is x or p, we do
    # <xi pi rj rk> = <(I + zi^2 - xi^2/2 - pi^2/2)rjrk>
    #   = i Lambda1[rj,rk] - Lambda2[i,i,rj,rk]/2 - Lambda2[n+i,n+i,rj,rk]/2 + <zi^2 rj rk>
    for i in range(n):
        Lambda2[i, n + i, :, :] = (
            1j * Lambda1
            - Lambda2[i, i, :, :] / 2.0
            - Lambda2[n + i, n + i, :, :] / 2.0
            + np.einsum("i,ij,ik", xpsamples[:, i], quadsamples, quadsamples)
        )
        Lambda2[i, :, n + i, :] = Lambda2[i, n + i, :, :]
        Lambda2[i, :, :, n + i] = Lambda2[i, n + i, :, :]
        Lambda2[:, i, :, n + i] = Lambda2[i, n + i, :, :]
        Lambda2[:, :, i, n + i] = Lambda2[i, n + i, :, :]
        Lambda2[:, i, n + i, :] = Lambda2[i, n + i, :, :]

        # now we use <pi xi rj rk> = -2i <rj rk> + <xprr>
        Lambda2[n + i, i, :, :] = -2j * Lambda1 + Lambda2[i, n + i, :, :]
        Lambda2[n + i, :, i, :] = Lambda2[n + i, i, :, :]
        Lambda2[n + i, :, :, i] = Lambda2[n + i, i, :, :]
        Lambda2[:, n + i, :, i] = Lambda2[n + i, i, :, :]
        Lambda2[:, :, n + i, i] = Lambda2[n + i, i, :, :]
        Lambda2[:, n + i, i, :] = Lambda2[n + i, i, :, :]

    ## NEED TO FINISH. THIS IS CRAZY. FIGURE OUT IF THERE IS A BETTER WAY.
    # z^2 = (x^2 + p^2)/2 + (xp+px)/2
    # z^3 = 1/(2sqrt2) (x^3 + x p^2 + x^2 p+xpx + p x^2 + p^3 + pxp+p^2x)

    ## next we deal with xi xi pi rj
    # xi (i + z^2 - x^2+p^2/2) rj

    # next we deal with xi pi xi rj

    # next we deal with pi xi xi rj

    ## next we deal with xi pi pi rj

    # next we deal with pi xi pi rj

    # next we deal with pi pi xi rj

    ## next we deal with xi xi xi pi

    # next we deal with xi xi pi xi

    # next we deal with xi pi xi xi

    # next we deal with pi xi xi xi

    ## next we deal with pi pi pi xi

    # next we deal with pi pi xi pi

    # next we deal with pi xi pi pi

    # next we deal with xi pi pi pi

    ## next we deal with xi xi pi pi

    # next we deal with xi pi xi pi

    # next we deal with pi xi xi pi

    # next we deal with xi pi pi xi

    # next we deal with pi xi pi xi

    # next we deal with pi pi xi xi

    Lambda2 = Lambda2.reshape(((2 * n) ** 2, (2 * n) ** 2))
    return Lambda1, Lambda2


# <psi|x p |psi>
# =
# -i\int y y barpsi(y) psi'(y)


# (adag + a)(adag - a)

# adag^2 - a^2 + 1

# ops.MeasureHeterodyne


# (cos x + sin p)(cos x + sin p)

# 1/2 x^2 + 1/2 p^2 + 1/2 {x, p}
"""
