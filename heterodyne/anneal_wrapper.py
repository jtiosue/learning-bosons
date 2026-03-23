from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import numpy as np


_ROOT = Path(__file__).resolve().parent
_DEFAULT_LIB = _ROOT / "libanneal.so"


class _ComplexDouble(ctypes.Structure):
    _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]


_ComplexDoublePtr = ctypes.POINTER(_ComplexDouble)


def _build_default_library(lib_path: Path) -> None:
    command = [
        "gcc",
        "-std=c11",
        "-O3",
        "-shared",
        "-fPIC",
        "anneal.c",
        "overlap.c",
        "random.c",
        "pcg_basic.c",
        "-lm",
        "-o",
        str(lib_path),
    ]
    try:
        subprocess.run(command, cwd=_ROOT, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        details = exc.stderr.strip() or exc.stdout.strip() or "unknown compiler error"
        raise RuntimeError(
            "Failed to build libanneal.so from the current C sources. "
            "The wrapper is ready, but the existing anneal sources must compile first.\n\n"
            f"Compiler output:\n{details}"
        ) from exc


def load_library(
    lib_path: str | Path | None = None, *, build_if_missing: bool = True
) -> ctypes.CDLL:
    path = Path(lib_path) if lib_path is not None else _DEFAULT_LIB
    if not path.is_absolute():
        path = (_ROOT / path).resolve()

    if not path.exists():
        if not build_if_missing:
            raise FileNotFoundError(f"Shared library not found: {path}")
        _build_default_library(path)

    library = ctypes.CDLL(str(path))
    library.sample_heterodyne.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(_ComplexDoublePtr),
        ctypes.POINTER(_ComplexDoublePtr),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        _ComplexDoublePtr,
        ctypes.POINTER(_ComplexDoublePtr),
    ]
    library.sample_heterodyne.restype = None

    library.overlap.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(_ComplexDoublePtr),
        ctypes.POINTER(_ComplexDoublePtr),
        ctypes.POINTER(ctypes.c_double),
        _ComplexDoublePtr,
        ctypes.POINTER(ctypes.c_double),
    ]
    library.overlap.restype = None

    return library


def _as_int_vector(values: np.ndarray | list[int], n: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.int32)
    if array.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {array.shape}")
    return np.ascontiguousarray(array)


def _as_float_vector(values: np.ndarray | list[float], n: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {array.shape}")
    return np.ascontiguousarray(array)


def _as_complex_vector(
    values: np.ndarray | list[complex], n: int, name: str
) -> np.ndarray:
    array = np.asarray(values, dtype=np.complex128)
    if array.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {array.shape}")
    return np.ascontiguousarray(array)


def _as_complex_matrix(values: np.ndarray, n: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.complex128)
    if array.shape != (n, n):
        raise ValueError(f"{name} must have shape ({n}, {n}), got {array.shape}")
    return np.ascontiguousarray(array)


def _matrix_row_pointers(matrix: np.ndarray) -> ctypes.Array:
    row_pointer_array = (_ComplexDoublePtr * matrix.shape[0])()
    stride = matrix.strides[0]
    base_addr = matrix.ctypes.data

    for row in range(matrix.shape[0]):
        row_pointer_array[row] = ctypes.cast(
            base_addr + row * stride, _ComplexDoublePtr
        )

    return row_pointer_array


def overlap(
    l,
    m,
    n,
    U,
    Up,
    ls,
    alpha,
    *,
    library: ctypes.CDLL | None = None,
    lib_path: str | Path | None = None,
):

    m_array = _as_int_vector(m, l, "m")
    n_array = _as_int_vector(n, l, "n")
    u_array = _as_complex_matrix(U, l, "U")
    p_array = _as_complex_matrix(Up, l, "P")
    ls_array = _as_float_vector(ls, l, "ls")
    alpha_array = _as_complex_vector(alpha, l, "alpha")

    lib = library if library is not None else load_library(lib_path)

    res = np.empty(1, dtype=np.float64)
    res_array = _as_float_vector(res, 1, "res")

    lib.overlap(
        l,
        m_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        n_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        _matrix_row_pointers(u_array),
        _matrix_row_pointers(p_array),
        ls_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        alpha_array.ctypes.data_as(_ComplexDoublePtr),
        res_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    return res[0]


def sample_heterodyne(
    n: int,
    f: np.ndarray | list[int],
    U: np.ndarray,
    P: np.ndarray,
    ls: np.ndarray | list[float],
    nsamples: int,
    stepsize: float,
    initial_anneal: int,
    delta: int,
    initial_alpha: np.ndarray | list[complex],
    *,
    library: ctypes.CDLL | None = None,
    lib_path: str | Path | None = None,
) -> np.ndarray:
    n_int = int(n)
    nsamples_int = int(nsamples)
    initial_anneal_int = int(initial_anneal)
    delta_int = int(delta)

    if n_int <= 0:
        raise ValueError("n must be positive")
    if nsamples_int < 0:
        raise ValueError("nsamples must be non-negative")
    if delta_int <= 0:
        raise ValueError("delta must be positive")

    lib = library if library is not None else load_library(lib_path)

    f_array = _as_int_vector(f, n_int, "f")
    u_array = _as_complex_matrix(U, n_int, "U")
    p_array = _as_complex_matrix(P, n_int, "P")
    ls_array = _as_float_vector(ls, n_int, "ls")
    alpha_array = _as_complex_vector(initial_alpha, n_int, "initial_alpha")
    output = np.empty((nsamples_int, n_int), dtype=np.complex128)

    lib.sample_heterodyne(
        n_int,
        f_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        _matrix_row_pointers(u_array),
        _matrix_row_pointers(p_array),
        ls_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        nsamples_int,
        stepsize,
        initial_anneal_int,
        delta_int,
        alpha_array.ctypes.data_as(_ComplexDoublePtr),
        _matrix_row_pointers(output),
    )

    return output
