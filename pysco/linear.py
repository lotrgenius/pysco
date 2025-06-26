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
    h2 = np.float32(1.0 / ncells_1d**2)
    qh2 = q * h2
    invsix = np.float32(1.0 / 6)
    result = np.empty_like(x)
    for i in prange(1, ncells_1d - 1):
        for j in prange(1, ncells_1d - 1):
            for k in prange(1, ncells_1d - 1):
                p = h2 * b[i, j, k] - invsix * (
                    x[i-1, j, k] + x[i+1, j, k] +
                    x[i, j-1, k] + x[i, j+1, k] +
                    x[i, j, k-1] + x[i, j, k+1]
                )
                result[i, j, k] = qh2 * x[i, j, k] + p
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
    """
    Linear residual with RHS

    Solve qu + p = rhs\
    with:\
    p = h^2*b - 1/6 * (u_{i+1,j,k} + u_{i-1,j,k} + u_{i,j+1,k} + u_{i,j-1,k} + u_{i,j,k+1} + u_{i,j,k-1})

    Computes residual = -(qu + p) + rhs

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    q : np.float32
        Linear coefficient on u
    rhs : npt.NDArray[np.float32]
        RHS term [N_cells_1d, N_cells_1d, N_cells_1d]

    Returns
    -------
    npt.NDArray[np.float32]
        Residual [N_cells_1d, N_cells_1d, N_cells_1d]

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.linear import residual_with_rhs
    >>> x = np.random.rand(32, 32, 32).astype(np.float32)
    >>> b = np.random.rand(32, 32, 32).astype(np.float32)
    >>> rhs = np.random.rand(32, 32, 32).astype(np.float32)
    >>> q = 0.05
    >>> result = residual_with_rhs(x, b, q, rhs)
    """
    ncells_1d = x.shape[0]
    h2 = np.float32(1.0 / ncells_1d**2)
    qh2 = q * h2
    invsix = np.float32(1.0 / 6)
    result = np.empty_like(x)
    for i in prange(1, ncells_1d - 1):
        for j in prange(1, ncells_1d - 1):
            for k in prange(1, ncells_1d - 1):
                p = h2 * b[i, j, k] - invsix * (
                    x[i-1, j, k] + x[i+1, j, k] +
                    x[i, j-1, k] + x[i, j+1, k] +
                    x[i, j, k-1] + x[i, j, k+1]
                )
                result[i, j, k] = -(qh2 * x[i, j, k] + p) + rhs[i, j, k]
    return result

@njit(["f4(f4, f4)"], fastmath=True, cache=True)
def solution_linear_equation(p: np.float32, q: np.float32) -> np.float32:
    """
    Linear solution of qu + p = 0

    Parameters
    ----------
    p : np.float32
        Constant term p (including stencil and source contribution)
    q : np.float32
        Coefficient of u

    Returns
    -------
    np.float32
        Solution of the linear equation

    Examples
    --------
    >>> from pysco.linear import solution_cubic_equation
    >>> p = 0.1
    >>> q = 0.05
    >>> solution = solution_cubic_equation(p, q)
    """
    return -p / q if q != 0 else np.float32(0)

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
    """
    Initialise potential for linear equation qu + p = 0

    Assumes initial stencil contribution is zero, so:
    p = h^2 * b, and u = -p / q = -h^2 * b / q

    Parameters
    ----------
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    q : np.float32
        Coefficient of u in the linear equation

    Returns
    -------
    npt.NDArray[np.float32]
        Initialised potential field

    Examples
    --------
    >>> import numpy as np
    >>> from pysco.linear import initialise_potential
    >>> b = np.random.rand(32, 32, 32).astype(np.float32)
    >>> q = -0.01
    >>> potential = initialise_potential(b, q)
    """
    h2 = np.float32(1.0 / b.shape[0]**2)
    result = np.empty_like(b)
    for i in prange(b.size):
        result.flat[i] = -h2 * b.flat[i] / ( h2 * q ) if q != 0 else np.float32(0)
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
    """
    Gauss-Seidel linear equation solver
    Solve qu + p = 0 using red-black ordering

    p = h^2 * b - 1/6 * (sum of 6-point stencil of u)

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    q : np.float32
        Coefficient on u
    f_relax : np.float32
        Relaxation factor
    """
    invsix = np.float32(1.0 / 6)
    ncells_1d = x.shape[0]
    ncells_1d_coarse = ncells_1d // 2
    h2 = np.float32(1.0 / ncells_1d**2)

    for parity in (0, 1):  # red-black order
        for i in prange(x.shape[0] >> 1):
            for j in prange(ncells_1d_coarse):
                for k in prange(ncells_1d_coarse):
                    if (i + j + k) % 2 == parity:
                        stencil = (
                            x[i-1, j, k] + x[i+1, j, k] +
                            x[i, j-1, k] + x[i, j+1, k] +
                            x[i, j, k-1] + x[i, j, k+1]
                        )
                        p = h2 * b[i, j, k] - invsix * stencil
                        x_new = solution_linear_equation(p, q)
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
    """
    Gauss-Seidel linear equation solver with RHS
    Solves qu + p = rhs using red-black ordering

    p = h^2 * b - 1/6 * (sum of 6-point stencil of u)

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Field [N, N, N]
    b : npt.NDArray[np.float32]
        Density field
    q : np.float32
        Coefficient of u
    rhs : npt.NDArray[np.float32]
        Right-hand side values
    f_relax : np.float32
        Relaxation factor
    """
    h2 = np.float32(1.0 / x.shape[0]**2)
    invsix = np.float32(1.0 / 6)

    for parity in (0, 1):  # red-black ordering
        for i in prange(1, x.shape[0] - 1):
            for j in prange(1, x.shape[0] - 1):
                for k in prange(1, x.shape[0] - 1):
                    if (i + j + k) % 2 == parity:
                        stencil = (
                            x[i-1, j, k] + x[i+1, j, k] +
                            x[i, j-1, k] + x[i, j+1, k] +
                            x[i, j, k-1] + x[i, j, k+1]
                        )
                        p = h2 * b[i, j, k] - invsix * stencil
                        full_p = p - rhs[i, j, k]
                        x_new = solution_linear_equation(full_p, q)
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
    """
    Residual of the linear operator on half the mesh
    Computes residual = -(qu + p), where:
    p = h^2 * b - 1/6 * (sum of 6-point stencil of u)

    This computes the residual only on alternating (red) points, assuming
    Gauss-Seidel was run without over-relaxation.

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential field
    b : npt.NDArray[np.float32]
        Density/source term
    q : np.float32
        Coefficient of u

    Returns
    -------
    npt.NDArray[np.float32]
        Residual field with values only on red sites

    Example
    -------
    >>> import numpy as np
    >>> from pysco.linear import residual_half
    >>> x = np.zeros((32, 32, 32), dtype=np.float32)
    >>> b = np.ones((32, 32, 32), dtype=np.float32)
    >>> q = 1.0
    >>> residual = residual_half(x, b, q)
    """
    invsix = np.float32(1.0 / 6)
    ncells_1d = x.shape[0]
    ncells_1d_coarse = ncells_1d // 2
    h2 = np.float32(1.0 / ncells_1d**2)
    qh2 = q * h2
    result = np.zeros_like(x)

    for i in prange(1, ncells_1d_coarse - 1):
        ii = 2 * i
        for j in prange(1, ncells_1d_coarse - 1):
            jj = 2 * j
            for k in prange(1, ncells_1d_coarse - 1):
                kk = 2 * k

                stencil_000 = (
                    x[ii-1, jj,   kk] +
                    x[ii+1, jj,   kk] +
                    x[ii,   jj-1, kk] +
                    x[ii,   jj+1, kk] +
                    x[ii,   jj,   kk-1] +
                    x[ii,   jj,   kk+1]
                )
                p = h2 * b[ii, jj, kk] - invsix * stencil_000
                result[ii, jj, kk] = -(qh2 * x[ii, jj, kk] + p)

    return result

@njit(["f4(f4[:,:,::1], f4[:,:,::1], f4)"], fastmath=True, cache=True, parallel=True)
def residual_error_half(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    q: np.float32,
) -> np.float32:
    """
    Error on half of the residual of the linear operator
    residual = qu + p, where:
    p = h^2 * b - 1/6 * (sum of 6-point stencil of u)

    error = sqrt[sum(residual**2)] over red sites only

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N, N, N]
    b : npt.NDArray[np.float32]
        Source/density field
    q : np.float32
        Coefficient of u

    Returns
    -------
    np.float32
        RMS residual error on red points

    Example
    -------
    >>> import numpy as np
    >>> from pysco.linear import residual_error_half
    >>> x = np.zeros((32, 32, 32), dtype=np.float32)
    >>> b = np.ones((32, 32, 32), dtype=np.float32)
    >>> q = 1.0
    >>> error = residual_error_half(x, b, q)
    """
    invsix = np.float32(1.0 / 6)
    h2 = np.float32(1.0 / x.shape[0]**2)
    qh2 = q * h2
    result = np.float32(0.0)

    for i in prange(1, x.shape[0]//2 - 1):  # red sites
        for j in prange(1, x.shape[1]//2 - 1):
            for k in prange(1, x.shape[2]//2 - 1):
                stencil = (
                    x[i-1, j, k] + x[i+1, j, k] +
                    x[i, j-1, k] + x[i, j+1, k] +
                    x[i, j, k-1] + x[i, j, k+1]
                )
                p = h2 * b[i, j, k] - invsix * stencil
                res = qh2 * x[i, j, k] + p
                result += res * res

    return np.sqrt(result)

@njit(["f4(f4[:,:,::1], f4[:,:,::1], f4)"], fastmath=True, cache=True, parallel=True)
def residual_error(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    q: np.float32
) -> np.float32:
    """
    Error on the residual of the linear operator
    residual = qu + p, where:
    p = h^2 * b - 1/6 * (sum of 6-point stencil of u)

    Computes total RMS error over the entire grid

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential [N, N, N]
    b : npt.NDArray[np.float32]
        Source term
    q : np.float32
        Coefficient of u

    Returns
    -------
    np.float32
        Total residual error (RMS)

    Example
    -------
    >>> import numpy as np
    >>> from pysco.linear import residual_error
    >>> x = np.zeros((32, 32, 32), dtype=np.float32)
    >>> b = np.ones((32, 32, 32), dtype=np.float32)
    >>> q = 1.0
    >>> error = residual_error(x, b, q)
    """
    invsix = np.float32(1.0 / 6)
    h2 = np.float32(1.0 / x.shape[0]**2)
    qh2 = q * h2
    result = np.float32(0)

    for i in prange(1, x.shape[0] - 1):
        for j in prange(1, x.shape[1] - 1):
            for k in prange(1, x.shape[2] - 1):
                stencil = (
                    x[i-1, j, k] + x[i+1, j, k] +
                    x[i, j-1, k] + x[i, j+1, k] +
                    x[i, j, k-1] + x[i, j, k+1]
                )
                p = h2 * b[i, j, k] - invsix * stencil
                res = qh2 * x[i, j, k] + p
                result += res * res

    return np.sqrt(result)

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
    """
    Restriction of residual of the linear operator on half the mesh.
    residual = -(qu + p), where:
    p = h^2 * b - 1/6 * (sum of 6-point stencil of u)

    This computes the residual on red points and restricts them to the coarser grid.

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential field [N, N, N]
    b : npt.NDArray[np.float32]
        Density/source term
    q : np.float32
        Coefficient on u

    Returns
    -------
    npt.NDArray[np.float32]
        Restricted residual on coarse red points [N//2, N//2, N//2]

    Example
    -------
    >>> import numpy as np
    >>> from pysco.linear import restrict_residual_half
    >>> x = np.zeros((32, 32, 32), dtype=np.float32)
    >>> b = np.ones((32, 32, 32), dtype=np.float32)
    >>> q = 1.0
    >>> r = restrict_residual_half(x, b, q)
    """
    invsix = np.float32(1.0 / 6)
    inveight = np.float32(1.0 / 8)
    ncells_1d = x.shape[0]
    h2 = np.float32(1.0 / ncells_1d**2)
    qh2 = q * h2
    ncells_1d_coarse = ncells_1d // 2

    result = np.empty(
        (ncells_1d_coarse, ncells_1d_coarse, ncells_1d_coarse), dtype=np.float32
    )

    for i in prange(1, ncells_1d_coarse - 1):
        ii = 2 * i
        for j in prange(1, ncells_1d_coarse - 1):
            jj = 2 * j
            for k in prange(1, ncells_1d_coarse - 1):
                kk = 2 * k

                # Compute 4 residuals from surrounding fine-grid red points
                res = np.float32(0.0)

                for di, dj, dk in [(0, 0, 0), (1, 0, 1), (1, 1, 0), (0, 1, 1)]:
                    iii, jjj, kkk = ii - di, jj - dj, kk - dk
                    stencil = (
                        x[iii-1, jjj,   kkk] +
                        x[iii+1, jjj,   kkk] +
                        x[iii,   jjj-1, kkk] +
                        x[iii,   jjj+1, kkk] +
                        x[iii,   jjj,   kkk-1] +
                        x[iii,   jjj,   kkk+1]
                    )
                    p = h2 * b[iii, jjj, kkk] - invsix * stencil
                    residual = -(qh2 * x[iii, jjj, kkk] + p)
                    res += residual

                result[i, j, k] = inveight * res  # average of 4 values

    return result

@njit(
    ["f4(f4[:,:,::1], f4[:,:,::1], f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def truncation_error(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    q: np.float32,
) -> np.float32:
    """
    Truncation error estimator for linear solver

    Estimates truncation error as:
    t = operator(restriction(fine)) - restriction(operator(fine))
    where operator(u) = qu + p, with p computed using the stencil

    Returns the RMS error from the difference

    Parameters
    ----------
    x : npt.NDArray[np.float32]
        Potential field on fine grid [N, N, N]
    b : npt.NDArray[np.float32]
        Source term on fine grid
    q : np.float32
        Coefficient of u in the linear operator

    Returns
    -------
    np.float32
        Truncation error estimate (RMS over coarse grid)

    Example
    -------
    >>> import numpy as np
    >>> from pysco.linear import truncation_error
    >>> x = np.zeros((32, 32, 32), dtype=np.float32)
    >>> b = np.ones((32, 32, 32), dtype=np.float32)
    >>> q = 1.0
    >>> error = truncation_error(x, b, q)
    """
    factor = np.float32(4.0)
    # Apply operator, then restrict vs restrict-then-operator
    restricted_operator = mesh.restriction(operator(x, b, q))
    operator_on_restricted = operator(mesh.restriction(x), mesh.restriction(b), q)

    r1 = restricted_operator.ravel()
    r2 = operator_on_restricted.ravel()
    size = r1.shape[0]

    result = np.float32(0.0)
    for i in prange(size):
        delta = factor * r1[i] - r2[i]
        result += delta * delta

    return np.sqrt(result)

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


