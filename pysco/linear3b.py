"""
Linear Operator Solver Module with Custom Biharmonic Stencil

Solves:
    c1 * Δ^2 u + c2 * Δ u + c3 * u + c4 = 0
or with a source term:
    c1 * Δ^2 u + c2 * Δ u + c3 * u + c4 = rhs

Inputs/Mapping:
  - x : field u, shape (N,N,N), float32
  - b : spatial field c4(x), shape (N,N,N), float32
  - q : float32[3] where q[0]=c1 (biharmonic), q[1]=c2 (Laplacian), q[2]=c3 (on u)

Operator form used everywhere (after multiplying PDE by h^2):
  qeff_h2 = c3*h^2 - 6*c2 + 42*c1/h^2
  p = h^2*c4 + c2*sum_faces(u) + (c1/h^2)*(-12*sum_faces + sum_axis2 + 2*sum_face_diag)
  operator(x) = qeff_h2 * x + p
"""

import numpy as np
import numpy.typing as npt
from numba import njit, prange
import mesh
import math


# ---------- Utility to compute stencil sums ----------
@njit(inline="always")
def _stencil_sums(x, i, j, k):
    """
    Returns (faces, axis2, fdiag) at index (i,j,k) using periodic indexing
    faces: (±1,0,0),(0,±1,0),(0,0,±1)     -> 6 points
    axis2: (±2,0,0),(0,±2,0),(0,0,±2)     -> 6 points
    fdiag: (±1,±1,0),(±1,0,±1),(0,±1,±1)  -> 12 points
    """
    im2 = i - 2; im1 = i - 1; ip1 = i + 1; ip2 = i + 2
    jm2 = j - 2; jm1 = j - 1; jp1 = j + 1; jp2 = j + 2
    km2 = k - 2; km1 = k - 1; kp1 = k + 1; kp2 = k + 2

    faces = (
        x[im1, j,   k  ] + x[ip1, j,   k  ] +
        x[i,   jm1, k  ] + x[i,   jp1, k  ] +
        x[i,   j,   km1] + x[i,   j,   kp1]
    )

    axis2 = (
        x[im2, j,   k  ] + x[ip2, j,   k  ] +
        x[i,   jm2, k  ] + x[i,   jp2, k  ] +
        x[i,   j,   km2] + x[i,   j,   kp2]
    )

    fdiag = (
        x[ip1, jp1, k  ] + x[im1, jp1, k  ] + x[ip1, jm1, k  ] + x[im1, jm1, k  ] +
        x[ip1, j,   kp1] + x[im1, j,   kp1] + x[ip1, j,   km1] + x[im1, j,   km1] +
        x[i,   jp1, kp1] + x[i,   jm1, kp1] + x[i,   jp1, km1] + x[i,   jm1, km1]
    )
    return faces, axis2, fdiag


# ---------- Core operator/residuals ----------

@njit(
    ["f4[:,:,::1](f4[:,:,::1], f4[:,:,::1], f4[::1])"],
    fastmath=True, cache=True, parallel=True,
)
def operator(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    q: npt.NDArray[np.float32],  # q[0]=c1, q[1]=c2, q[2]=c3
) -> npt.NDArray[np.float32]:
    """
    Returns (qeff*h^2)*x + p with the custom Δ^2 stencil.
    """
    n = x.shape[0]
    h2 = np.float32(1.0 / (n * n))
    invh2 = np.float32(n * n)

    c1 = q[0]; c2 = q[1]; c3 = q[2]
    qeff_h2 = np.float32(c3 * h2 - np.float32(6.0) * c2 + np.float32(42.0) * c1 * invh2)

    out = np.empty_like(x)

    # support ±2 neighbors via periodic indexing by using [-2, n-2)
    for i in prange(-2, n - 2):
        for j in prange(-2, n - 2):
            for k in prange(-2, n - 2):
                faces, axis2, fdiag = _stencil_sums(x, i, j, k)

                p = h2 * b[i, j, k]
                p += c2 * faces
                p += c1 * invh2 * (-np.float32(12.0) * faces + axis2 + np.float32(2.0) * fdiag)

                out[i, j, k] = qeff_h2 * x[i, j, k] + p
    return out


@njit(
    ["f4[:,:,::1](f4[:,:,::1], f4[:,:,::1], f4[::1], f4[:,:,::1])"],
    fastmath=True, cache=True, parallel=True,
)
def residual_with_rhs(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    q: npt.NDArray[np.float32],
    rhs: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """
    residual = rhs - ((qeff*h^2)*x + p)
    """
    n = x.shape[0]
    h2 = np.float32(1.0 / (n * n))
    invh2 = np.float32(n * n)

    c1 = q[0]; c2 = q[1]; c3 = q[2]
    qeff_h2 = np.float32(c3 * h2 - np.float32(6.0) * c2 + np.float32(42.0) * c1 * invh2)

    out = np.empty_like(x)
    for i in prange(-2, n - 2):
        for j in prange(-2, n - 2):
            for k in prange(-2, n - 2):
                faces, axis2, fdiag = _stencil_sums(x, i, j, k)
                p = h2 * b[i, j, k]
                p += c2 * faces
                p += c1 * invh2 * (-np.float32(12.0) * faces + axis2 + np.float32(2.0) * fdiag)
                out[i, j, k] = rhs[i, j, k] - (qeff_h2 * x[i, j, k] + p)
    return out


# ---------- Local update helper (signature preserved) ----------

@njit(["f4(f4, f4)"], fastmath=True, cache=True)
def solution_linear_equation(p_like: np.float32, d1_like: np.float32) -> np.float32:
    """
    Reused helper with same signature.
    Interprets:
        target u* = - p_like / (d1_like/27)
    Guard tiny denominator: if |d1_like| < eps -> return 0.
    """
    denom = np.float64(d1_like) / 27.0
    eps = 1e-30
    if abs(denom) < eps:
        return np.float32(0.0)
    return np.float32(-np.float64(p_like) / denom)


# ---------- Initialization / Smoothing (GS) ----------

@njit(
    ["f4[:,:,::1](f4[:,:,::1], f4[::1])"],
    fastmath=True, cache=True, parallel=True,
)
def initialise_potential(
    b: npt.NDArray[np.float32],
    q: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """
    Initialize u using neighbors≈0 approximation:
      faces≈0, axis2≈0, fdiag≈0  => p ≈ h^2*c4
      qeff*h^2 as usual.
      u ≈ -p / (qeff*h^2)  with tiny-denominator guard.
    """
    n = b.shape[0]
    h2 = np.float32(1.0 / (n * n))
    invh2 = np.float32(n * n)

    c1 = q[0]; c2 = q[1]; c3 = q[2]
    qeff_h2 = np.float32(c3 * h2 - np.float32(6.0) * c2 + np.float32(42.0) * c1 * invh2)

    d1_like = np.float32(27.0) * qeff_h2  # so helper returns -p/(qeff_h2)
    u = np.empty_like(b)
    u_r = u.ravel()
    b_r = b.ravel()
    size = len(b_r)
    for idx in prange(size):
        p_like = h2 * b_r[idx]
        u_r[idx] = solution_linear_equation(p_like, d1_like)
    return u


@njit(
    ["void(f4[:,:,::1], f4[:,:,::1], f4[::1], f4)"],
    fastmath=True, cache=True, parallel=True,
)
def gauss_seidel(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    q: npt.NDArray[np.float32],
    f_relax: np.float32,
) -> None:
    """
    Red-Black Gauss-Seidel for:
      (qeff*h^2) u + p = 0
    Update target at each visited point:
      u* = -p / (qeff*h^2), relaxed with factor f_relax.
    """
    n = x.shape[0]
    h2 = np.float32(1.0 / (n * n))
    invh2 = np.float32(n * n)

    c1 = q[0]; c2 = q[1]; c3 = q[2]
    qeff_h2 = np.float32(c3 * h2 - np.float32(6.0) * c2 + np.float32(42.0) * c1 * invh2)
    d1_like = np.float32(27.0) * qeff_h2  # helper denominator scale

    # "Red" sweep: visit (i-1,j-1,k-1), (i-1,j,k), (i,j-1,k), (i,j,k-1) pattern
    # We'll explicitly update four interleaved points per coarse cell, as in original structure.
    n_coarse = n // 2

    # Red
    for i in prange(n // 2):
        ii = 2 * i
        for j in prange(n_coarse):
            jj = 2 * j
            for k in prange(n_coarse):
                kk = 2 * k

                # (ii-1, jj-1, kk-1)
                i0 = ii - 1; j0 = jj - 1; k0 = kk - 1
                f1, a2, fd = _stencil_sums(x, i0, j0, k0)
                p_like = h2 * b[i0, j0, k0]
                p_like += c2 * f1
                p_like += c1 * invh2 * (-np.float32(12.0) * f1 + a2 + np.float32(2.0) * fd)
                x[i0, j0, k0] += f_relax * (solution_linear_equation(p_like, d1_like) - x[i0, j0, k0])

                # (ii-1, jj, kk)
                i1 = ii - 1; j1 = jj; k1 = kk
                f1, a2, fd = _stencil_sums(x, i1, j1, k1)
                p_like = h2 * b[i1, j1, k1] + c2 * f1 + c1 * invh2 * (-np.float32(12.0) * f1 + a2 + np.float32(2.0) * fd)
                x[i1, j1, k1] += f_relax * (solution_linear_equation(p_like, d1_like) - x[i1, j1, k1])

                # (ii, jj-1, kk)
                i2 = ii; j2 = jj - 1; k2 = kk
                f1, a2, fd = _stencil_sums(x, i2, j2, k2)
                p_like = h2 * b[i2, j2, k2] + c2 * f1 + c1 * invh2 * (-np.float32(12.0) * f1 + a2 + np.float32(2.0) * fd)
                x[i2, j2, k2] += f_relax * (solution_linear_equation(p_like, d1_like) - x[i2, j2, k2])

                # (ii, jj, kk-1)
                i3 = ii; j3 = jj; k3 = kk - 1
                f1, a2, fd = _stencil_sums(x, i3, j3, k3)
                p_like = h2 * b[i3, j3, k3] + c2 * f1 + c1 * invh2 * (-np.float32(12.0) * f1 + a2 + np.float32(2.0) * fd)
                x[i3, j3, k3] += f_relax * (solution_linear_equation(p_like, d1_like) - x[i3, j3, k3])

    # Black
    for i in prange(n_coarse):
        ii = 2 * i
        for j in prange(n_coarse):
            jj = 2 * j
            for k in prange(n_coarse):
                kk = 2 * k

                # (ii-1, jj-1, kk)
                i0 = ii - 1; j0 = jj - 1; k0 = kk
                f1, a2, fd = _stencil_sums(x, i0, j0, k0)
                p_like = h2 * b[i0, j0, k0] + c2 * f1 + c1 * invh2 * (-np.float32(12.0) * f1 + a2 + np.float32(2.0) * fd)
                x[i0, j0, k0] += f_relax * (solution_linear_equation(p_like, d1_like) - x[i0, j0, k0])

                # (ii-1, jj, kk-1)
                i1 = ii - 1; j1 = jj; k1 = kk - 1
                f1, a2, fd = _stencil_sums(x, i1, j1, k1)
                p_like = h2 * b[i1, j1, k1] + c2 * f1 + c1 * invh2 * (-np.float32(12.0) * f1 + a2 + np.float32(2.0) * fd)
                x[i1, j1, k1] += f_relax * (solution_linear_equation(p_like, d1_like) - x[i1, j1, k1])

                # (ii, jj-1, kk-1)
                i2 = ii; j2 = jj - 1; k2 = kk - 1
                f1, a2, fd = _stencil_sums(x, i2, j2, k2)
                p_like = h2 * b[i2, j2, k2] + c2 * f1 + c1 * invh2 * (-np.float32(12.0) * f1 + a2 + np.float32(2.0) * fd)
                x[i2, j2, k2] += f_relax * (solution_linear_equation(p_like, d1_like) - x[i2, j2, k2])

                # (ii, jj, kk)
                i3 = ii; j3 = jj; k3 = kk
                f1, a2, fd = _stencil_sums(x, i3, j3, k3)
                p_like = h2 * b[i3, j3, k3] + c2 * f1 + c1 * invh2 * (-np.float32(12.0) * f1 + a2 + np.float32(2.0) * fd)
                x[i3, j3, k3] += f_relax * (solution_linear_equation(p_like, d1_like) - x[i3, j3, k3])


@njit(
    ["void(f4[:,:,::1], f4[:,:,::1], f4[::1], f4[:,:,::1], f4)"],
    fastmath=True, cache=True, parallel=True,
)
def gauss_seidel_with_rhs(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    q: npt.NDArray[np.float32],
    rhs: npt.NDArray[np.float32],
    f_relax: np.float32,
) -> None:
    """
    Red-Black GS for:
      (qeff*h^2) u + p = rhs
    Update target:
      u* = (rhs - p)/(qeff*h^2)
    """
    n = x.shape[0]
    h2 = np.float32(1.0 / (n * n))
    invh2 = np.float32(n * n)

    c1 = q[0]; c2 = q[1]; c3 = q[2]
    qeff_h2 = np.float32(c3 * h2 - np.float32(6.0) * c2 + np.float32(42.0) * c1 * invh2)
    d1_like = np.float32(27.0) * qeff_h2

    n_coarse = n // 2

    # Red
    for i in prange(n // 2):
        ii = 2 * i
        for j in prange(n_coarse):
            jj = 2 * j
            for k in prange(n_coarse):
                kk = 2 * k

                # (ii-1, jj-1, kk-1)
                i0 = ii - 1; j0 = jj - 1; k0 = kk - 1
                f1, a2, fd = _stencil_sums(x, i0, j0, k0)
                p_like = h2 * b[i0, j0, k0] + c2 * f1 + c1 * invh2 * (-np.float32(12.0) * f1 + a2 + np.float32(2.0) * fd)
                p_eff = p_like - rhs[i0, j0, k0]  # so helper returns -(p_eff)/(qeff_h2) = (rhs - p)/(qeff_h2)
                x[i0, j0, k0] += f_relax * (solution_linear_equation(p_eff, d1_like) - x[i0, j0, k0])

                # (ii-1, jj, kk)
                i1 = ii - 1; j1 = jj; k1 = kk
                f1, a2, fd = _stencil_sums(x, i1, j1, k1)
                p_like = h2 * b[i1, j1, k1] + c2 * f1 + c1 * invh2 * (-np.float32(12.0) * f1 + a2 + np.float32(2.0) * fd)
                p_eff = p_like - rhs[i1, j1, k1]
                x[i1, j1, k1] += f_relax * (solution_linear_equation(p_eff, d1_like) - x[i1, j1, k1])

                # (ii, jj-1, kk)
                i2 = ii; j2 = jj - 1; k2 = kk
                f1, a2, fd = _stencil_sums(x, i2, j2, k2)
                p_like = h2 * b[i2, j2, k2] + c2 * f1 + c1 * invh2 * (-np.float32(12.0) * f1 + a2 + np.float32(2.0) * fd)
                p_eff = p_like - rhs[i2, j2, k2]
                x[i2, j2, k2] += f_relax * (solution_linear_equation(p_eff, d1_like) - x[i2, j2, k2])

                # (ii, jj, kk-1)
                i3 = ii; j3 = jj; k3 = kk - 1
                f1, a2, fd = _stencil_sums(x, i3, j3, k3)
                p_like = h2 * b[i3, j3, k3] + c2 * f1 + c1 * invh2 * (-np.float32(12.0) * f1 + a2 + np.float32(2.0) * fd)
                p_eff = p_like - rhs[i3, j3, k3]
                x[i3, j3, k3] += f_relax * (solution_linear_equation(p_eff, d1_like) - x[i3, j3, k3])

    # Black
    for i in prange(n_coarse):
        ii = 2 * i
        for j in prange(n_coarse):
            jj = 2 * j
            for k in prange(n_coarse):
                kk = 2 * k

                # (ii-1, jj-1, kk)
                i0 = ii - 1; j0 = jj - 1; k0 = kk
                f1, a2, fd = _stencil_sums(x, i0, j0, k0)
                p_like = h2 * b[i0, j0, k0] + c2 * f1 + c1 * invh2 * (-np.float32(12.0) * f1 + a2 + np.float32(2.0) * fd)
                p_eff = p_like - rhs[i0, j0, k0]
                x[i0, j0, k0] += f_relax * (solution_linear_equation(p_eff, d1_like) - x[i0, j0, k0])

                # (ii-1, jj, kk-1)
                i1 = ii - 1; j1 = jj; k1 = kk - 1
                f1, a2, fd = _stencil_sums(x, i1, j1, k1)
                p_like = h2 * b[i1, j1, k1] + c2 * f1 + c1 * invh2 * (-np.float32(12.0) * f1 + a2 + np.float32(2.0) * fd)
                p_eff = p_like - rhs[i1, j1, k1]
                x[i1, j1, k1] += f_relax * (solution_linear_equation(p_eff, d1_like) - x[i1, j1, k1])

                # (ii, jj-1, kk-1)
                i2 = ii; j2 = jj - 1; k2 = kk - 1
                f1, a2, fd = _stencil_sums(x, i2, j2, k2)
                p_like = h2 * b[i2, j2, k2] + c2 * f1 + c1 * invh2 * (-np.float32(12.0) * f1 + a2 + np.float32(2.0) * fd)
                p_eff = p_like - rhs[i2, j2, k2]
                x[i2, j2, k2] += f_relax * (solution_linear_equation(p_eff, d1_like) - x[i2, j2, k2])

                # (ii, jj, kk)
                i3 = ii; j3 = jj; k3 = kk
                f1, a2, fd = _stencil_sums(x, i3, j3, k3)
                p_like = h2 * b[i3, j3, k3] + c2 * f1 + c1 * invh2 * (-np.float32(12.0) * f1 + a2 + np.float32(2.0) * fd)
                p_eff = p_like - rhs[i3, j3, k3]
                x[i3, j3, k3] += f_relax * (solution_linear_equation(p_eff, d1_like) - x[i3, j3, k3])


# ---------- Residuals / Errors ----------

@njit(
    ["f4[:,:,::1](f4[:,:,::1], f4[:,:,::1], f4[::1])"],
    fastmath=True, cache=True, parallel=True,
)
def residual_half(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    q: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """
    residual on half-mesh (red points), no RHS:
      residual = -((qeff*h^2)*u + p)
    """
    n = x.shape[0]
    h2 = np.float32(1.0 / (n * n))
    invh2 = np.float32(n * n)

    c1 = q[0]; c2 = q[1]; c3 = q[2]
    qeff_h2 = np.float32(c3 * h2 - np.float32(6.0) * c2 + np.float32(42.0) * c1 * invh2)

    out = np.zeros_like(x)
    n_coarse = n // 2

    for i in prange(n_coarse):
        ii = 2 * i
        for j in prange(n_coarse):
            jj = 2 * j
            for k in prange(n_coarse):
                kk = 2 * k

                # (ii-1, jj-1, kk-1)
                i0 = ii - 1; j0 = jj - 1; k0 = kk - 1
                f1, a2, fd = _stencil_sums(x, i0, j0, k0)
                p = h2 * b[i0, j0, k0] + c2 * f1 + c1 * invh2 * (-np.float32(12.0) * f1 + a2 + np.float32(2.0) * fd)
                out[i0, j0, k0] = -(qeff_h2 * x[i0, j0, k0] + p)

                # (ii-1, jj, kk)
                i1 = ii - 1; j1 = jj; k1 = kk
                f1, a2, fd = _stencil_sums(x, i1, j1, k1)
                p = h2 * b[i1, j1, k1] + c2 * f1 + c1 * invh2 * (-np.float32(12.0) * f1 + a2 + np.float32(2.0) * fd)
                out[i1, j1, k1] = -(qeff_h2 * x[i1, j1, k1] + p)

                # (ii, jj-1, kk)
                i2 = ii; j2 = jj - 1; k2 = kk
                f1, a2, fd = _stencil_sums(x, i2, j2, k2)
                p = h2 * b[i2, j2, k2] + c2 * f1 + c1 * invh2 * (-np.float32(12.0) * f1 + a2 + np.float32(2.0) * fd)
                out[i2, j2, k2] = -(qeff_h2 * x[i2, j2, k2] + p)

                # (ii, jj, kk-1)
                i3 = ii; j3 = jj; k3 = kk - 1
                f1, a2, fd = _stencil_sums(x, i3, j3, k3)
                p = h2 * b[i3, j3, k3] + c2 * f1 + c1 * invh2 * (-np.float32(12.0) * f1 + a2 + np.float32(2.0) * fd)
                out[i3, j3, k3] = -(qeff_h2 * x[i3, j3, k3] + p)
    return out


@njit(["f4(f4[:,:,::1], f4[:,:,::1], f4[::1])"], fastmath=True, cache=True, parallel=True)
def residual_error_half(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    q: npt.NDArray[np.float32],
) -> np.float32:
    """
    error over red half:
      err = sqrt(sum(residual^2)), residual = (qeff*h^2)*u + p
    """
    n = x.shape[0]
    h2 = np.float32(1.0 / (n * n))
    invh2 = np.float32(n * n)

    c1 = q[0]; c2 = q[1]; c3 = q[2]
    qeff_h2 = np.float32(c3 * h2 - np.float32(6.0) * c2 + np.float32(42.0) * c1 * invh2)

    res = np.float32(0.0)
    n_coarse = n // 2

    for i in prange(n_coarse):
        ii = 2 * i
        for j in prange(n_coarse):
            jj = 2 * j
            for k in prange(n_coarse):
                kk = 2 * k

                # (ii-1, jj-1, kk-1)
                i0 = ii - 1; j0 = jj - 1; k0 = kk - 1
                f1, a2, fd = _stencil_sums(x, i0, j0, k0)
                p = h2 * b[i0, j0, k0] + c2 * f1 + c1 * invh2 * (-np.float32(12.0) * f1 + a2 + np.float32(2.0) * fd)
                r = qeff_h2 * x[i0, j0, k0] + p
                res += r * r

                # (ii-1, jj, kk)
                i1 = ii - 1; j1 = jj; k1 = kk
                f1, a2, fd = _stencil_sums(x, i1, j1, k1)
                p = h2 * b[i1, j1, k1] + c2 * f1 + c1 * invh2 * (-np.float32(12.0) * f1 + a2 + np.float32(2.0) * fd)
                r = qeff_h2 * x[i1, j1, k1] + p
                res += r * r

                # (ii, jj-1, kk)
                i2 = ii; j2 = jj - 1; k2 = kk
                f1, a2, fd = _stencil_sums(x, i2, j2, k2)
                p = h2 * b[i2, j2, k2] + c2 * f1 + c1 * invh2 * (-np.float32(12.0) * f1 + a2 + np.float32(2.0) * fd)
                r = qeff_h2 * x[i2, j2, k2] + p
                res += r * r

                # (ii, jj, kk-1)
                i3 = ii; j3 = jj; k3 = kk - 1
                f1, a2, fd = _stencil_sums(x, i3, j3, k3)
                p = h2 * b[i3, j3, k3] + c2 * f1 + c1 * invh2 * (-np.float32(12.0) * f1 + a2 + np.float32(2.0) * fd)
                r = qeff_h2 * x[i3, j3, k3] + p
                res += r * r

    return np.sqrt(res)


@njit(["f4(f4[:,:,::1], f4[:,:,::1], f4[::1])"], fastmath=True, cache=True, parallel=True)
def residual_error(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    q: npt.NDArray[np.float32],
) -> np.float32:
    """
    error over full mesh:
      err = sqrt(sum( ((qeff*h^2)*u + p)^2 ))
    """
    n = x.shape[0]
    h2 = np.float32(1.0 / (n * n))
    invh2 = np.float32(n * n)

    c1 = q[0]; c2 = q[1]; c3 = q[2]
    qeff_h2 = np.float32(c3 * h2 - np.float32(6.0) * c2 + np.float32(42.0) * c1 * invh2)

    res = np.float32(0.0)
    for i in prange(-2, n - 2):
        for j in prange(-2, n - 2):
            for k in prange(-2, n - 2):
                f1, a2, fd = _stencil_sums(x, i, j, k)
                p = h2 * b[i, j, k] + c2 * f1 + c1 * invh2 * (-np.float32(12.0) * f1 + a2 + np.float32(2.0) * fd)
                r = qeff_h2 * x[i, j, k] + p
                res += r * r
    return np.sqrt(res)


@njit(
    ["f4[:,:,::1](f4[:,:,::1], f4[:,:,::1], f4[::1])"],
    fastmath=True, cache=True, parallel=True,
)
def restrict_residual_half(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    q: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """
    Restrict residual over red half (no RHS):
      residual = -((qeff*h^2)*u + p)
    Then standard half-weighting (here simple average of the 4 red points).
    """
    n = x.shape[0]
    h2 = np.float32(1.0 / (n * n))
    invh2 = np.float32(n * n)

    c1 = q[0]; c2 = q[1]; c3 = q[2]
    qeff_h2 = np.float32(c3 * h2 - np.float32(6.0) * c2 + np.float32(42.0) * c1 * invh2)

    n_coarse = n // 2
    result = np.empty((n_coarse, n_coarse, n_coarse), dtype=np.float32)
    inveight = np.float32(0.125)

    for i in prange(n_coarse):
        ii = 2 * i
        for j in prange(n_coarse):
            jj = 2 * j
            for k in prange(n_coarse):
                kk = 2 * k

                # 4 red points around coarse cell
                acc = np.float32(0.0)

                i0 = ii - 1; j0 = jj - 1; k0 = kk - 1
                f1, a2, fd = _stencil_sums(x, i0, j0, k0)
                p = h2 * b[i0, j0, k0] + c2 * f1 + c1 * invh2 * (-np.float32(12.0) * f1 + a2 + np.float32(2.0) * fd)
                acc += -(qeff_h2 * x[i0, j0, k0] + p)

                i1 = ii - 1; j1 = jj; k1 = kk
                f1, a2, fd = _stencil_sums(x, i1, j1, k1)
                p = h2 * b[i1, j1, k1] + c2 * f1 + c1 * invh2 * (-np.float32(12.0) * f1 + a2 + np.float32(2.0) * fd)
                acc += -(qeff_h2 * x[i1, j1, k1] + p)

                i2 = ii; j2 = jj - 1; k2 = kk
                f1, a2, fd = _stencil_sums(x, i2, j2, k2)
                p = h2 * b[i2, j2, k2] + c2 * f1 + c1 * invh2 * (-np.float32(12.0) * f1 + a2 + np.float32(2.0) * fd)
                acc += -(qeff_h2 * x[i2, j2, k2] + p)

                i3 = ii; j3 = jj; k3 = kk - 1
                f1, a2, fd = _stencil_sums(x, i3, j3, k3)
                p = h2 * b[i3, j3, k3] + c2 * f1 + c1 * invh2 * (-np.float32(12.0) * f1 + a2 + np.float32(2.0) * fd)
                acc += -(qeff_h2 * x[i3, j3, k3] + p)

                result[i, j, k] = inveight * acc
    return result


@njit(
    ["f4(f4[:,:,::1], f4[:,:,::1], f4[::1])"],
    fastmath=True, cache=True, parallel=True,
)
def truncation_error(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    q: npt.NDArray[np.float32],
) -> np.float32:
    """
    Truncation error estimator (NR-style):
      t = Operator(Restriction(x)) - Restriction(Operator(x))
      terr = sqrt( sum (4*RLx - LRx)^2 )   # factor 4 for grid correction
    """
    four = np.float32(4)
    RLx = mesh.restriction(operator(x, b, q))
    LRx = operator(mesh.restriction(x), mesh.restriction(b), q)
    RLx_ravel = RLx.ravel()
    LRx_ravel = LRx.ravel()
    size = len(RLx_ravel)
    acc = np.float32(0.0)
    for i in prange(size):
        acc += (four * RLx_ravel[i] - LRx_ravel[i]) ** 2
    return np.sqrt(acc)


# ---------- Simple smoothing wrappers ----------

def smoothing(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    q: npt.NDArray[np.float32],
    n_smoothing: int,
) -> None:
    """
    Run several Gauss-Seidel iterations (no RHS).
    """
    f_relax = np.float32(1.25)
    for _ in range(n_smoothing):
        gauss_seidel(x, b, q, f_relax)


def smoothing_with_rhs(
    x: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    q: npt.NDArray[np.float32],
    n_smoothing: int,
    rhs: npt.NDArray[np.float32],
) -> None:
    """
    Run several Gauss-Seidel iterations (with RHS).
    """
    f_relax = np.float32(1.25)
    for _ in range(n_smoothing):
        gauss_seidel_with_rhs(x, b, q, rhs, f_relax)
