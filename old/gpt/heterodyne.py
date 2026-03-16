from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from thewalrus.symplectic import expand, is_symplectic


HBAR = 2.0
DEFAULT_CHOI_R = float(np.arcsinh(1.0))
DEFAULT_MAX_TENSOR_ENTRIES = 2_000_000


def _as_fock_tuple(f: Sequence[int] | NDArray[np.integer]) -> tuple[int, ...]:
    values = tuple(int(x) for x in f)
    if not values:
        raise ValueError("f must contain at least one mode")
    if any(x < 0 for x in values):
        raise ValueError("f must contain only non-negative integers")
    return values


def _validate_symplectic(S: ArrayLike, n: int, check_symplectic: bool) -> NDArray[np.float64]:
    matrix = np.asarray(S, dtype=float)
    if matrix.shape != (2 * n, 2 * n):
        raise ValueError(f"S must have shape {(2 * n, 2 * n)}, got {matrix.shape}")
    if check_symplectic and not is_symplectic(matrix):
        raise ValueError("S is not symplectic")
    return matrix


def _complex_displacements_from_xp(mu: NDArray[np.float64], hbar: float = HBAR) -> NDArray[np.complex128]:
    n_modes = len(mu) // 2
    alpha = (mu[:n_modes] + 1j * mu[n_modes:]) / math.sqrt(2.0 * hbar)
    return np.concatenate([alpha, alpha.conj()]).astype(np.complex128, copy=False)


def _qmat_from_cov(cov: NDArray[np.float64], hbar: float = HBAR) -> NDArray[np.complex128]:
    n_modes = cov.shape[0] // 2
    identity = np.identity(n_modes)

    x = cov[:n_modes, :n_modes] * (2.0 / hbar)
    xp = cov[:n_modes, n_modes:] * (2.0 / hbar)
    p = cov[n_modes:, n_modes:] * (2.0 / hbar)

    aidaj = (x + p + 1j * (xp - xp.T) - 2.0 * identity) / 4.0
    aiaj = (x - p + 1j * (xp + xp.T)) / 4.0

    return np.block([[aidaj, aiaj.conj()], [aiaj, aidaj.conj()]]) + np.identity(2 * n_modes)


def _amat_from_cov(cov: NDArray[np.float64], hbar: float = HBAR) -> NDArray[np.complex128]:
    n_modes = cov.shape[0] // 2
    identity = np.identity(2 * n_modes)
    exchange = np.block(
        [[np.zeros((n_modes, n_modes)), np.identity(n_modes)], [np.identity(n_modes), np.zeros((n_modes, n_modes))]]
    )
    qmat = _qmat_from_cov(cov, hbar=hbar)
    return exchange @ (identity - np.linalg.inv(qmat)).conj()


def _sqrt_table(cutoff: int) -> NDArray[np.float64]:
    return np.sqrt(np.arange(cutoff + 1, dtype=float))


def _hafnian_batched_renorm(
    matrix: NDArray[np.complex128],
    displacement: NDArray[np.complex128],
    cutoff: int,
) -> NDArray[np.complex128]:
    shape = (cutoff,) * displacement.size
    values = np.zeros(shape, dtype=np.complex128)
    values[(0,) * displacement.size] = 1.0
    sqrt_values = _sqrt_table(cutoff)

    iterator = np.ndindex(shape)
    next(iterator)
    for index in iterator:
        leading_axis = next(axis for axis, value in enumerate(index) if value > 0)
        previous = list(index)
        previous[leading_axis] -= 1
        previous_tuple = tuple(previous)

        term = displacement[leading_axis] * values[previous_tuple]
        for axis, count in enumerate(previous_tuple):
            if count == 0:
                continue
            reduced = list(previous_tuple)
            reduced[axis] -= 1
            term += sqrt_values[count] * matrix[leading_axis, axis] * values[tuple(reduced)]

        values[index] = term / sqrt_values[index[leading_axis]]

    return values


def _gaussian_state_vector(
    mu: NDArray[np.float64],
    cov: NDArray[np.float64],
    cutoff: int,
    *,
    choi_r: float | None = None,
) -> NDArray[np.complex128]:
    beta = _complex_displacements_from_xp(mu, hbar=HBAR)
    amat = _amat_from_cov(cov, hbar=HBAR)

    n_total_modes = mu.size // 2
    bmat = amat[:n_total_modes, :n_total_modes]
    alpha = beta[:n_total_modes]
    gamma = alpha.conj() - bmat @ alpha
    prefactor = np.exp(np.conj(-0.5 * (np.linalg.norm(alpha) ** 2 - alpha @ bmat @ alpha)))

    if choi_r is not None:
        half_modes = n_total_modes // 2
        scaling = np.concatenate(
            [np.ones(half_modes), np.full(half_modes, 1.0 / math.tanh(choi_r), dtype=float)]
        )
        bmat = np.diag(scaling) @ bmat @ np.diag(scaling)
        gamma = scaling * gamma

    psi = _hafnian_batched_renorm(bmat.conj(), gamma.conj(), cutoff)
    return prefactor * psi


def _required_tensor_entries(n_modes: int, cutoff: int) -> int:
    return cutoff ** (2 * n_modes)


def _gaussian_unitary_tensor(
    S: NDArray[np.float64],
    cutoff: int,
    *,
    choi_r: float,
    max_tensor_entries: int,
) -> NDArray[np.complex128]:
    n_modes = S.shape[0] // 2
    required_entries = _required_tensor_entries(n_modes, cutoff)
    if required_entries > max_tensor_entries:
        raise ValueError(
            "The requested cutoff is too large for the Choi tensor construction. "
            f"Need {required_entries} entries, but max_tensor_entries={max_tensor_entries}."
        )

    cosh_r = math.cosh(choi_r)
    sinh_r = math.sinh(choi_r)
    identity = np.identity(n_modes)
    zeros = np.zeros((n_modes, n_modes))
    choi_symplectic = np.block(
        [
            [cosh_r * identity, sinh_r * identity, zeros, zeros],
            [sinh_r * identity, cosh_r * identity, zeros, zeros],
            [zeros, zeros, cosh_r * identity, -sinh_r * identity],
            [zeros, zeros, -sinh_r * identity, cosh_r * identity],
        ]
    )
    expanded = expand(S, list(range(n_modes)), 2 * n_modes) @ choi_symplectic
    covariance = expanded @ expanded.T
    means = np.zeros(4 * n_modes, dtype=float)
    return _gaussian_state_vector(means, covariance, cutoff, choi_r=choi_r)


def _output_state(
    S: NDArray[np.float64],
    f: tuple[int, ...],
    cutoff: int,
    *,
    choi_r: float,
    max_tensor_entries: int,
) -> tuple[NDArray[np.complex128], float]:
    if max(f) >= cutoff:
        raise ValueError("cutoff must be strictly larger than every entry of f")

    tensor = _gaussian_unitary_tensor(
        S,
        cutoff,
        choi_r=choi_r,
        max_tensor_entries=max_tensor_entries,
    )
    index = (slice(None),) * len(f) + tuple(f)
    state = np.asarray(tensor[index], dtype=np.complex128)
    weight = float(np.linalg.norm(state.ravel()) ** 2)
    if weight <= 0.0:
        raise ValueError("The truncated output state vanished. Increase cutoff.")
    return state / math.sqrt(weight), weight


def _coherent_overlap(state: NDArray[np.complex128], alpha: NDArray[np.complex128]) -> complex:
    polynomial = state
    cutoff = state.shape[0]

    for mode_alpha in alpha:
        basis = np.empty(cutoff, dtype=np.complex128)
        basis[0] = 1.0
        conjugate = np.conj(mode_alpha)
        for photon_number in range(1, cutoff):
            basis[photon_number] = basis[photon_number - 1] * conjugate / math.sqrt(photon_number)
        polynomial = np.tensordot(polynomial, basis, axes=([0], [0]))

    gaussian = math.exp(-0.5 * float(np.vdot(alpha, alpha).real))
    return complex(gaussian * polynomial)


def _log_husimi_density(state: NDArray[np.complex128], alpha: NDArray[np.complex128]) -> float:
    overlap = _coherent_overlap(state, alpha)
    magnitude = abs(overlap)
    if magnitude == 0.0:
        return -math.inf
    n_modes = state.ndim
    return 2.0 * math.log(magnitude) - n_modes * math.log(math.pi)


def _input_covariance(f: tuple[int, ...]) -> NDArray[np.float64]:
    variances = np.repeat(2.0 * np.asarray(f, dtype=float) + 1.0, 2)
    return np.diag(variances)


def _heterodyne_variances(S: NDArray[np.float64], f: tuple[int, ...]) -> NDArray[np.float64]:
    n_modes = len(f)
    covariance = S @ _input_covariance(f) @ S.T
    x_var = np.diag(covariance[:n_modes, :n_modes])
    p_var = np.diag(covariance[n_modes:, n_modes:])
    return np.maximum((x_var + p_var + 2.0) / 4.0, 1e-12)


def _proposal_logpdf(alpha: NDArray[np.complex128], variances: NDArray[np.float64]) -> float:
    return float(-np.sum(np.abs(alpha) ** 2 / variances + np.log(math.pi * variances)))


def _proposal_sample(rng: np.random.Generator, variances: NDArray[np.float64]) -> NDArray[np.complex128]:
    scales = np.sqrt(variances / 2.0)
    return scales * (rng.standard_normal(variances.size) + 1j * rng.standard_normal(variances.size))


def sample_heterodyne(
    S: ArrayLike,
    f: Sequence[int] | NDArray[np.integer],
    nsamples: int,
    **params,
) -> NDArray[np.complex128]:
    """Sample approximate heterodyne outcomes for U_S |f>.

    The returned samples come from the Husimi Q function of the cutoff-truncated
    state obtained by applying the Gaussian unitary defined by S to the Fock state
    |f>. Exact sampling is impossible in general because the output has infinite
    Fock support, so this routine makes two controlled approximations:

    1. The Gaussian unitary is represented in a Fock basis with a finite cutoff.
    2. Samples are drawn with an independence Metropolis-Hastings chain targeting
       the truncated Husimi density.

    Keyword parameters:
        cutoff:
            Fock cutoff used to build the truncated state. Default is
            max(max(f) + 4, 8).
        burn_in:
            Number of initial MH steps to discard. Default is 500.
        thinning:
            Number of MH steps between returned samples. Default is 10.
        proposal_scale:
            Positive scalar or per-mode iterable multiplying the default proposal
            variances inferred from second moments. Default is 1.0.
        seed:
            Optional RNG seed.
        initial_state:
            Optional initial point for the MH chain as a length-n complex vector.
        choi_r:
            Two-mode squeezing parameter used in the Choi expansion. Default is
            arcsinh(1).
        check_symplectic:
            If true, validate S before sampling. Default is true.
        max_tensor_entries:
            Hard cap on cutoff**(2n) for the internal Choi tensor. Default is
            2_000_000.

    Returns:
        An array of shape (nsamples, n_modes) with complex heterodyne samples.
    """
    if nsamples < 0:
        raise ValueError("nsamples must be non-negative")

    fock = _as_fock_tuple(f)
    n_modes = len(fock)
    cutoff = int(params.get("cutoff", max(max(fock) + 4, 8)))
    burn_in = int(params.get("burn_in", 500))
    thinning = int(params.get("thinning", 10))
    choi_r = float(params.get("choi_r", DEFAULT_CHOI_R))
    check_symplectic = bool(params.get("check_symplectic", True))
    max_tensor_entries = int(params.get("max_tensor_entries", DEFAULT_MAX_TENSOR_ENTRIES))

    if cutoff < 1:
        raise ValueError("cutoff must be positive")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative")
    if thinning < 1:
        raise ValueError("thinning must be at least 1")

    symplectic = _validate_symplectic(S, n_modes, check_symplectic)
    truncated_state, truncation_weight = _output_state(
        symplectic,
        fock,
        cutoff,
        choi_r=choi_r,
        max_tensor_entries=max_tensor_entries,
    )

    if truncation_weight < 0.9:
        raise ValueError(
            f"Cutoff {cutoff} captures only {truncation_weight:.3f} of the output norm. Increase cutoff."
        )

    if nsamples == 0:
        return np.empty((0, n_modes), dtype=np.complex128)

    raw_scale = params.get("proposal_scale", 1.0)
    base_variances = _heterodyne_variances(symplectic, fock)
    if isinstance(raw_scale, Iterable) and not np.isscalar(raw_scale):
        scale = np.asarray(tuple(raw_scale), dtype=float)
        if scale.shape != (n_modes,):
            raise ValueError(f"proposal_scale must have shape {(n_modes,)} when iterable")
    else:
        scale = np.full(n_modes, float(raw_scale), dtype=float)
    if np.any(scale <= 0.0):
        raise ValueError("proposal_scale must be strictly positive")
    proposal_variances = base_variances * scale

    rng = np.random.default_rng(params.get("seed", None))

    initial_state = params.get("initial_state")
    if initial_state is None:
        current = _proposal_sample(rng, proposal_variances)
    else:
        current = np.asarray(initial_state, dtype=np.complex128)
        if current.shape != (n_modes,):
            raise ValueError(f"initial_state must have shape {(n_modes,)}")

    current_log_target = _log_husimi_density(truncated_state, current)
    current_log_proposal = _proposal_logpdf(current, proposal_variances)
    attempts = 0
    while not math.isfinite(current_log_target):
        if attempts >= 32:
            raise RuntimeError("Failed to find a finite-probability initial state")
        current = _proposal_sample(rng, proposal_variances)
        current_log_target = _log_husimi_density(truncated_state, current)
        current_log_proposal = _proposal_logpdf(current, proposal_variances)
        attempts += 1

    samples = np.empty((nsamples, n_modes), dtype=np.complex128)
    total_steps = burn_in + nsamples * thinning
    saved = 0

    for step in range(total_steps):
        proposal = _proposal_sample(rng, proposal_variances)
        proposal_log_target = _log_husimi_density(truncated_state, proposal)
        proposal_log_proposal = _proposal_logpdf(proposal, proposal_variances)

        log_acceptance = (
            proposal_log_target
            + current_log_proposal
            - current_log_target
            - proposal_log_proposal
        )
        if math.log(rng.random()) < min(0.0, log_acceptance):
            current = proposal
            current_log_target = proposal_log_target
            current_log_proposal = proposal_log_proposal

        if step >= burn_in and (step - burn_in) % thinning == 0:
            samples[saved] = current
            saved += 1

    return samples


__all__ = ["sample_heterodyne"]