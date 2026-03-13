import numpy as np
import strawberryfields as sf
from strawberryfields.ops import Fock, GaussianTransform, MeasureHomodyne

def sample_homodyne_from_fock_gaussian(
    n_occupations,
    S,
    nsamples,
    cutoff_dim=15,
    phis=None
):
    """
    Sample homodyne outcomes from U_S |n1,...,nm>.

    Args:
        n_occupations: list/tuple of length m, e.g. [1,0,2]
        S: (2m, 2m) real symplectic matrix
        nsamples: number of samples to generate
        cutoff_dim: Fock cutoff dimension
        phis: list of homodyne angles, one per mode; default all 0 (x quadrature)

    Returns:
        samples: array of shape (nsamples, m)
    """
    n_occupations = list(n_occupations)
    m = len(n_occupations)

    S = np.array(S, dtype=float)
    if S.shape != (2*m, 2*m):
        raise ValueError(f"S must have shape {(2*m, 2*m)}")

    if phis is None:
        phis = [0.0] * m
    if len(phis) != m:
        raise ValueError("phis must have length m")

    samples = np.zeros((nsamples, m), dtype=float)

    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})

    for k in range(nsamples):
        prog = sf.Program(m)

        with prog.context as q:
            # Prepare |n1,...,nm>
            for i, ni in enumerate(n_occupations):
                Fock(ni) | q[i]

            # Apply the Gaussian unitary corresponding to S
            GaussianTransform(S) | q

            # Homodyne each mode
            for i, phi in enumerate(phis):
                MeasureHomodyne(phi=phi) | q[i]

        result = eng.run(prog)
        # result.samples has shape (1, m) for one shot
        samples[k, :] = result.samples[0]

        eng.reset()

    return samples
