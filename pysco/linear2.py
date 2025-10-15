import numpy as np
import numpy.typing as npt
from numba import config, njit, prange
import mesh

#NEW FOR MOG

@njit([
    "f4[:,:,::1](f4[:,:,::1], f4[:,:,::1], f4)"
], fastmath=True, cache=True, parallel=True)
def operator(x: npt.NDArray[np.float32], b: npt.NDArray[np.float32], q: np.float32) -> npt.NDArray[np.float32]:
    """
    Linear operator

    Solve qu + p = 0\
    with:\
    p = h^2*b - 1/6 * (u_{i+1,j,k} + u_{i-1,j,k} + u_{i,j+1,k} + u_{i,j-1,k} + u_{i,j,k+1} + u_{i,j,k-1})

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    q : np.float32
        Linear coefficient on u

    Returns
    -------
    npt.NDArray[np.float32]
        Result of qu + p [N_cells_1d, N_cells_1d, N_cells_1d]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.linear import operator
    >>> x = np.random.rand(32, 32, 32).astype(np.float32)
    >>> b = np.random.rand(32, 32, 32).astype(np.float32)
    >>> q = 0.05
    >>> result = operator(x, b, q)
    """
    ncells_1d = x.shape[0]
    h4 = np.float32(1.0 / ncells_1d**4)
    qh4 = q * h4
    invfortytwo = np.float32(1.0 / 42)
    result = np.empty_like(x)
    for i in prange(1, ncells_1d - 1):
        for j in prange(1, ncells_1d - 1):
            for k in prange(1, ncells_1d - 1):
                p = h4 * b[i, j, k] + invfortytwo * (
                    -12 * (x[i+1, j, k] +
                    x[i, j+1, k] +
                    x[i, j, k+1] +
                    x[i-1, j, k] +
                    x[i, j-1, k] +
                    x[i, j, k-1]) +
                    x[i+2, j, k] +
                    x[i, j+2, k] +
                    x[i, j, k+2] +
                    x[i-2, j, k] +
                    x[i, j-2, k] +
                    x[i, j, k-2] +
                    2 * (x[i+1, j, k+1] +
                    x[i, j+1, k+1] +
                    x[i-1, j, k+1] +
                    x[i, j-1, k+1] +
                    x[i+1, j, k-1] +
                    x[i, j+1, k-1] +
                    x[i-1, j, k-1] +
                    x[i, j-1, k-1] +
                    x[i-1, j+1, k] +
                    x[i-1, j-1, k] +
                    x[i+1, j-1, k] +
                    x[i+1, j+1, k])
                )
                result[i, j, k] = qh4 * x[i, j, k] + p
    return result

@njit([
    "f4[:,:,::1](f4[:,:,::1], f4[:,:,::1], f4, f4[:,:,::1])"
], fastmath=True, cache=True, parallel=True)
def residual_with_rhs(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    q: np.float32,
    rhs: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    ncells_1d = x.shape[0]
    h4 = np.float32(1.0 / ncells_1d**4)
    qh4 = q * h4
    invfortytwo = np.float32(1.0 / 42)
    result = np.zeros_like(x)

    for i in prange(2, ncells_1d - 2):
        for j in prange(2, ncells_1d - 2):
            for k in prange(2, ncells_1d - 2):
                p = h4 * b[i, j, k] + invfortytwo * (
                    -12 * (x[i+1, j, k] + x[i-1, j, k] +
                           x[i, j+1, k] + x[i, j-1, k] +
                           x[i, j, k+1] + x[i, j, k-1]) +
                    (x[i+2, j, k] + x[i-2, j, k] +
                     x[i, j+2, k] + x[i, j-2, k] +
                     x[i, j, k+2] + x[i, j, k-2]) +
                    2 * (x[i+1, j+1, k] + x[i+1, j-1, k] +
                         x[i-1, j+1, k] + x[i-1, j-1, k] +
                         x[i+1, j, k+1] + x[i-1, j, k+1] +
                         x[i, j+1, k+1] + x[i, j-1, k+1] +
                         x[i+1, j, k-1] + x[i-1, j, k-1] +
                         x[i, j+1, k-1] + x[i, j-1, k-1])
                )
                result[i, j, k] = -(qh4 * x[i, j, k] + p) + rhs[i, j, k]
    return result

@njit(
    ["f4[:,:,::1](f4[:,:,::1], f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def initialise_potential(
    b: npt.NDArray[np.float32],
    q: np.float32,
) -> npt.NDArray[np.float32]:
    # With ∆² and initial stencil=0, q*h4*u + h4*b = 0  ⇒  u = -b/q
    result = np.empty_like(b)
    if q != 0:
        invq = np.float32(1.0) / q
        for idx in prange(b.size):
            result.flat[idx] = -b.flat[idx] * invq
    else:
        for idx in prange(b.size):
            result.flat[idx] = np.float32(0)
    return result

@njit(
    ["void(f4[:,:,::1], f4[:,:,::1], f4, f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def gauss_seidel(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    q: np.float32,
    f_relax: np.float32,
) -> None:
    ncells_1d = x.shape[0]
    h4 = np.float32(1.0 / ncells_1d**4)
    qh4 = q * h4
    invfortytwo = np.float32(1.0 / 42)

    for parity in (0, 1):
        for i in prange(2, ncells_1d - 2):
            for j in prange(2, ncells_1d - 2):
                for k in prange(2, ncells_1d - 2):
                    if (i + j + k) % 2 == parity:
                        bih = invfortytwo * (
                            -12 * (x[i+1, j, k] + x[i-1, j, k] +
                                   x[i, j+1, k] + x[i, j-1, k] +
                                   x[i, j, k+1] + x[i, j, k-1]) +
                            (x[i+2, j, k] + x[i-2, j, k] +
                             x[i, j+2, k] + x[i, j-2, k] +
                             x[i, j, k+2] + x[i, j, k-2]) +
                            2 * (x[i+1, j+1, k] + x[i+1, j-1, k] +
                                 x[i-1, j+1, k] + x[i-1, j-1, k] +
                                 x[i+1, j, k+1] + x[i-1, j, k+1] +
                                 x[i, j+1, k+1] + x[i, j-1, k+1] +
                                 x[i+1, j, k-1] + x[i-1, j, k-1] +
                                 x[i, j+1, k-1] + x[i, j-1, k-1])
                        )
                        p = h4 * b[i, j, k] + bih
                        # qh4*x + p = 0  ⇒  x_new = -p/qh4
                        x_new = -p / qh4 if qh4 != 0 else np.float32(0)
                        x[i, j, k] += f_relax * (x_new - x[i, j, k])

@njit(
    ["void(f4[:,:,::1], f4[:,:,::1], f4, f4[:,:,::1], f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def gauss_seidel_with_rhs(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    q: np.float32,
    rhs: npt.NDArray[np.float32],
    f_relax: np.float32,
) -> None:
    ncells_1d = x.shape[0]
    h4 = np.float32(1.0 / ncells_1d**4)
    qh4 = q * h4
    invfortytwo = np.float32(1.0 / 42)

    for parity in (0, 1):
        for i in prange(2, ncells_1d - 2):
            for j in prange(2, ncells_1d - 2):
                for k in prange(2, ncells_1d - 2):
                    if (i + j + k) % 2 == parity:
                        bih = invfortytwo * (
                            -12 * (x[i+1, j, k] + x[i-1, j, k] +
                                   x[i, j+1, k] + x[i, j-1, k] +
                                   x[i, j, k+1] + x[i, j, k-1]) +
                            (x[i+2, j, k] + x[i-2, j, k] +
                             x[i, j+2, k] + x[i, j-2, k] +
                             x[i, j, k+2] + x[i, j, k-2]) +
                            2 * (x[i+1, j+1, k] + x[i+1, j-1, k] +
                                 x[i-1, j+1, k] + x[i-1, j-1, k] +
                                 x[i+1, j, k+1] + x[i-1, j, k+1] +
                                 x[i, j+1, k+1] + x[i, j-1, k+1] +
                                 x[i+1, j, k-1] + x[i-1, j, k-1] +
                                 x[i, j+1, k-1] + x[i, j-1, k-1])
                        )
                        p = h4 * b[i, j, k] + bih
                        # qh4*x + p = rhs  ⇒  x_new = (rhs - p)/qh4
                        x_new = (rhs[i, j, k] - p) / qh4 if qh4 != 0 else np.float32(0)
                        x[i, j, k] += f_relax * (x_new - x[i, j, k])

@njit(
    ["f4[:,:,::1](f4[:,:,::1], f4[:,:,::1], f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def residual_half(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    q: np.float32,
) -> npt.NDArray[np.float32]:
    ncells_1d = x.shape[0]
    h4 = np.float32(1.0 / ncells_1d**4)
    qh4 = q * h4
    invfortytwo = np.float32(1.0 / 42)
    result = np.zeros_like(x)
    ncells_1d_coarse = ncells_1d // 2

    for i in prange(2, ncells_1d_coarse - 2):
        ii = 2 * i
        for j in prange(2, ncells_1d_coarse - 2):
            jj = 2 * j
            for k in prange(2, ncells_1d_coarse - 2):
                kk = 2 * k
                bih = invfortytwo * (
                    -12 * (x[ii+1, jj, kk] + x[ii-1, jj, kk] +
                           x[ii, jj+1, kk] + x[ii, jj-1, kk] +
                           x[ii, jj, kk+1] + x[ii, jj, kk-1]) +
                    (x[ii+2, jj, kk] + x[ii-2, jj, kk] +
                     x[ii, jj+2, kk] + x[ii, jj-2, kk] +
                     x[ii, jj, kk+2] + x[ii, jj, kk-2]) +
                    2 * (x[ii+1, jj+1, kk] + x[ii+1, jj-1, kk] +
                         x[ii-1, jj+1, kk] + x[ii-1, jj-1, kk] +
                         x[ii+1, jj, kk+1] + x[ii-1, jj, kk+1] +
                         x[ii, jj+1, kk+1] + x[ii, jj-1, kk+1] +
                         x[ii+1, jj, kk-1] + x[ii-1, jj, kk-1] +
                         x[ii, jj+1, kk-1] + x[ii, jj-1, kk-1])
                )
                p = h4 * b[ii, jj, kk] + bih
                result[ii, jj, kk] = -(qh4 * x[ii, jj, kk] + p)
    return result

@njit(["f4(f4[:,:,::1], f4[:,:,::1], f4)"], fastmath=True, cache=True, parallel=True)
def residual_error_half(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    q: np.float32,
) -> np.float32:
    ncells_1d = x.shape[0]
    h4 = np.float32(1.0 / ncells_1d**4)
    qh4 = q * h4
    invfortytwo = np.float32(1.0 / 42)
    acc = np.float32(0.0)

    # red sites on the full grid but obey the ±2 reach
    for i in prange(2, x.shape[0] - 2):
        for j in prange(2, x.shape[1] - 2):
            for k in prange(2, x.shape[2] - 2):
                if (i + j + k) % 2 == 0:
                    bih = invfortytwo * (
                        -12 * (x[i+1, j, k] + x[i-1, j, k] +
                               x[i, j+1, k] + x[i, j-1, k] +
                               x[i, j, k+1] + x[i, j, k-1]) +
                        (x[i+2, j, k] + x[i-2, j, k] +
                         x[i, j+2, k] + x[i, j-2, k] +
                         x[i, j, k+2] + x[i, j, k-2]) +
                        2 * (x[i+1, j+1, k] + x[i+1, j-1, k] +
                             x[i-1, j+1, k] + x[i-1, j-1, k] +
                             x[i+1, j, k+1] + x[i-1, j, k+1] +
                             x[i, j+1, k+1] + x[i, j-1, k+1] +
                             x[i+1, j, k-1] + x[i-1, j, k-1] +
                             x[i, j+1, k-1] + x[i, j-1, k-1])
                    )
                    p = h4 * b[i, j, k] + bih
                    res = qh4 * x[i, j, k] + p
                    acc += res * res
    return np.sqrt(acc)

@njit(["f4(f4[:,:,::1], f4[:,:,::1], f4)"], fastmath=True, cache=True, parallel=True)
def residual_error(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    q: np.float32
) -> np.float32:
    ncells_1d = x.shape[0]
    h4 = np.float32(1.0 / ncells_1d**4)
    qh4 = q * h4
    invfortytwo = np.float32(1.0 / 42)
    acc = np.float32(0.0)

    for i in prange(2, x.shape[0] - 2):
        for j in prange(2, x.shape[1] - 2):
            for k in prange(2, x.shape[2] - 2):
                bih = invfortytwo * (
                    -12 * (x[i+1, j, k] + x[i-1, j, k] +
                           x[i, j+1, k] + x[i, j-1, k] +
                           x[i, j, k+1] + x[i, j, k-1]) +
                    (x[i+2, j, k] + x[i-2, j, k] +
                     x[i, j+2, k] + x[i, j-2, k] +
                     x[i, j, k+2] + x[i, j, k-2]) +
                    2 * (x[i+1, j+1, k] + x[i+1, j-1, k] +
                         x[i-1, j+1, k] + x[i-1, j-1, k] +
                         x[i+1, j, k+1] + x[i-1, j, k+1] +
                         x[i, j+1, k+1] + x[i, j-1, k+1] +
                         x[i+1, j, k-1] + x[i-1, j, k-1] +
                         x[i, j+1, k-1] + x[i, j-1, k-1])
                )
                p = h4 * b[i, j, k] + bih
                res = qh4 * x[i, j, k] + p
                acc += res * res
    return np.sqrt(acc)

@njit(
    ["f4[:,:,::1](f4[:,:,::1], f4[:,:,::1], f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def restrict_residual_half(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    q: np.float32,
) -> npt.NDArray[np.float32]:
    ncells_1d = x.shape[0]
    h4 = np.float32(1.0 / ncells_1d**4)
    qh4 = q * h4
    invfortytwo = np.float32(1.0 / 42)
    inveight = np.float32(1.0 / 8)

    ncells_1d_coarse = ncells_1d // 2
    result = np.zeros((ncells_1d_coarse, ncells_1d_coarse, ncells_1d_coarse), dtype=np.float32)

    # Compute residuals on red points near the coarse nodes and average (same pattern as before, now with ∆²)
    for i in prange(2, ncells_1d_coarse - 2):
        ii = 2 * i
        for j in prange(2, ncells_1d_coarse - 2):
            jj = 2 * j
            for k in prange(2, ncells_1d_coarse - 2):
                kk = 2 * k

                res = np.float32(0.0)

                for di, dj, dk in [(0, 0, 0), (1, 0, 1), (1, 1, 0), (0, 1, 1)]:
                    iii, jjj, kkk = ii - di, jj - dj, kk - dk
                    stencil = (
                    -12 * (x[iii+1, jjj, kkk] +
                    x[iii, jjj+1, kkk] +
                    x[iii, jjj, kkk+1] +
                    x[iii-1, jjj, kkk] +
                    x[iii, jjj-1, kkk] +
                    x[iii, jjj, kkk-1]) +
                    x[iii+2, jjj, kkk] +
                    x[iii, jjj+2, kkk] +
                    x[iii, jjj, kkk+2] +
                    x[iii-2, jjj, kkk] +
                    x[iii, jjj-2, kkk] +
                    x[iii, jjj, kkk-2] +
                    2 * (x[iii+1, jjj, kkk+1] +
                    x[iii, jjj+1, kkk+1] +
                    x[iii-1, jjj, kkk+1] +
                    x[iii, jjj-1, kkk+1] +
                    x[iii+1, jjj, kkk-1] +
                    x[iii, jjj+1, kkk-1] +
                    x[iii-1, jjj, kkk-1] +
                    x[iii, jjj-1, kkk-1] +
                    x[iii-1, jjj+1, kkk] +
                    x[iii-1, jjj-1, kkk] +
                    x[iii+1, jjj-1, kkk] +
                    x[iii+1, jjj+1, kkk])
                    )
                    p = h4 * b[iii, jjj, kkk] - invfortytwo * stencil
                    residual = -(qh4 * x[iii, jjj, kkk] + p)
                    res += residual

                result[i, j, k] = inveight * res  # average of 4 values

    return result


def smoothing(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    q: np.float32,
    n_smoothing: int,
) -> None:
    """
    Smooth solution field with several Gauss-Seidel iterations for the linear operator

    This performs n_smoothing iterations of the linear Gauss-Seidel solver:
    solving qu + p = 0 using stencil-based relaxation.

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential field [N, N, N]
    b : npt.NDArray[np.float32]
        Source term
    q : np.float32
        Coefficient of u
    n_smoothing : int
        Number of Gauss-Seidel relaxation steps

    Example
    -------
    >>> import numpy as np
    >>> from pysco.linear import smoothing
    >>> x = np.zeros((32, 32, 32), dtype=np.float32)
    >>> b = np.ones((32, 32, 32), dtype=np.float32)
    >>> q = 1.0
    >>> smoothing(x, b, q, 5)
    """
    f_relax = np.float32(1.25)
    for _ in range(n_smoothing):
        gauss_seidel(x, b, q, f_relax)

def smoothing_with_rhs(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    q: np.float32,
    n_smoothing: int,
    rhs: npt.NDArray[np.float32],
) -> None:
    """
    Smooth field with Gauss-Seidel iterations with RHS for the linear operator

    Solves qu + p = rhs using several Gauss-Seidel sweeps with over-relaxation.
    Useful for multigrid smoothing steps.

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential field [N, N, N]
    b : npt.NDArray[np.float32]
        Source field
    q : np.float32
        Coefficient on u
    n_smoothing : int
        Number of smoothing iterations
    rhs : npt.NDArray[np.float32]
        Right-hand side of the equation [N, N, N]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.linear import smoothing_with_rhs
    >>> x = np.zeros((32, 32, 32), dtype=np.float32)
    >>> b = np.ones((32, 32, 32), dtype=np.float32)
    >>> q = 1.0
    >>> rhs = np.random.rand(32, 32, 32).astype(np.float32)
    >>> smoothing_with_rhs(x, b, q, 5, rhs)
    """
    f_relax = np.float32(1.25)
    for _ in range(n_smoothing):
        gauss_seidel_with_rhs(x, b, q, rhs, f_relax)